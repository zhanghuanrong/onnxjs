// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {InferenceHandler, SessionHandler} from '../../backend';
import {ExecutionPlan} from '../../execution-plan';
import {Graph} from '../../graph';
import {Operator} from '../../operators';
import {OpSet, resolveOperator} from '../../opset';
import {Session} from '../../session';
import {Tensor} from '../../tensor';
import {ProtoUtil} from '../../util';
import {getInstance, InferenceContext} from '../../wasm-binding-core';
import {WasmBackend} from '../backend-wasm';

// import {CPU_OP_RESOLVE_RULES} from '../cpu/op-resolve-rules';

import {WasmInferenceHandler} from './inference-handler';
import {OP_INFO_RESOLVE_RULES} from './op-resolve-rules-vnext';
import {OperatorInfo} from './op-vnext';

// import {WASM_OP_RESOLVE_RULES} from './op-resolve-rules';

export class WasmSessionHandler implements SessionHandler {
  customExecutionPlan: ExecutionPlan;
  wasmContext: InferenceContext;
  private bindingInstance: ReturnType<typeof getInstance>;

  // private opResolveRules: ReadonlyArray<OpSet.ResolveRule>;
  constructor(readonly backend: WasmBackend, readonly context: Session.Context, fallbackToCpuOps: boolean) {
    // this.opResolveRules = fallbackToCpuOps ? WASM_OP_RESOLVE_RULES.concat(CPU_OP_RESOLVE_RULES) :
    // WASM_OP_RESOLVE_RULES;
  }

  createInferenceHandler(): InferenceHandler {
    return new WasmInferenceHandler(this, this.context.profiler);
  }

  onGraphInitialized(graph: Graph, opset: ReadonlyArray<OpSet>) {
    // STEP.1 inference types
    const types = graph.getValueTypes().slice();
    const ops = graph.getNodes().map(node => resolveOperator<OperatorInfo>(node, opset, OP_INFO_RESOLVE_RULES));
    ops.forEach((op, i) => {
      const node = graph.getNodes()[i];
      const inputTypes = node.inputs.map(i => types[i]);

      let undefinedTypeIndex = inputTypes.indexOf(0);
      if (undefinedTypeIndex !== -1) {
        throw new Error(
            `type inference failed: unexpected undefined type at input[${undefinedTypeIndex}] of node ${node.name}`);
      }
      const inferencedOutputTypes = op.inferenceType(inputTypes);
      undefinedTypeIndex = inferencedOutputTypes.indexOf(0);
      if (undefinedTypeIndex !== -1) {
        throw new Error(
            `type inference failed: unexpected undefined type at output[${undefinedTypeIndex}] of node ${node.name}`);
      }
      if (inferencedOutputTypes.length !== node.outputs.length) {
        throw new Error(`type inference failed: length mismatch for output of node ${node.name}`);
      }
      inferencedOutputTypes.forEach((t, i) => {
        if (types[node.outputs[i]] !== 0 && types[node.outputs[i]] !== t) {
          throw new Error(
              `type inference failed: conflict inference result at output[${undefinedTypeIndex}] of node ${node.name}`);
        }
        types[node.outputs[i]] = t;
      });
    });

    // STEP.2 create inference context
    this.bindingInstance = getInstance();
    this.wasmContext =
        new (this.bindingInstance.InferenceContext)(graph.getNodes().length, graph.getValues().length, types);

    // STEP.3 set initializers
    graph.getValues().forEach((v, i) => {
      if (v.from === -1 && v.tensor) {
        this.wasmContext.setInitializer(i, v.tensor.dims);
        this.uploadData(v.tensor.numberData, this.wasmContext.getTensorData(i), this.wasmContext.getTensorDataSize(i));
      }
    });

    // STEP.4 set attributes
    ops.forEach((op, i) => {
      const attributes = graph.getNodes()[i].attributes;
      op.initializeAttributes(attributes);
      attributes.forEach((name, type, value) => {
        switch (type) {
          case 'float':
            this.wasmContext.addAttribute_f(i, name, value as number);
            break;
          case 'floats':
            this.wasmContext.addAttribute_floats(i, name, value as number[]);
            break;
          case 'int':
            this.wasmContext.addAttribute_i(i, name, safeInt32(value as number));
            break;
          case 'ints':
            this.wasmContext.addAttribute_ints(i, name, (value as number[]).map(safeInt32));
            break;
          case 'string':
            this.wasmContext.addAttribute_s(i, name, (value as string));
            break;
          case 'strings':
            this.wasmContext.addAttribute_strings(i, name, (value as string[]));
            break;
          default:
            throw new Error(`unsupported attribute type: ${type}`);
        }
      });
    });

    // STEP.5 init kernels
    ops.forEach((op, i) => {
      const node = graph.getNodes()[i];
      this.wasmContext.initKernel(
          i, node.opType, opset[0].domain, opset[0].version, node.inputs, node.outputs, op.hash);
    });

    // ExecutionPlan implementation
    this.customExecutionPlan = {
      execute(sessionHandler: WasmSessionHandler, modelInputs: Tensor[]): Promise<Tensor[]> {
        for (let i = 0; i < modelInputs.length; i++) {
          const valueIndex = graph.getInputIndices()[i];
          sessionHandler.wasmContext.setInput(valueIndex, modelInputs[i].dims);
          sessionHandler.uploadData(
              modelInputs[i].numberData, sessionHandler.wasmContext.getTensorData(valueIndex),
              sessionHandler.wasmContext.getTensorDataSize(valueIndex));
        }

        sessionHandler.wasmContext.run();

        const outputs = [];
        for (let i = 0; i < graph.getOutputIndices().length; i++) {
          const valueIndex = graph.getOutputIndices()[i];

          const tensor = new Tensor(
              sessionHandler.wasmContext.getTensorShape(valueIndex),
              ProtoUtil.tensorDataTypeFromProto(types[valueIndex]));
          sessionHandler.downloadData(
              tensor.numberData, sessionHandler.wasmContext.getTensorData(valueIndex),
              sessionHandler.wasmContext.getTensorDataSize(valueIndex));
          outputs.push(tensor);
        }
        return Promise.resolve(outputs);
      },

    };
  }

  uploadData(data: Tensor.NumberType, byteOffset: number, length: number): void {
    new (data.constructor as Float32ArrayConstructor)(this.bindingInstance.HEAPU8.buffer, byteOffset, length).set(data);
  }

  downloadData(data: Tensor.NumberType, byteOffset: number, length: number): void {
    data.set(new (data.constructor as Float32ArrayConstructor)(this.bindingInstance.HEAPU8.buffer, byteOffset, length));
  }

  dispose(): void {}

  resolve(node: Graph.Node, opsets: ReadonlyArray<OpSet>, graph: Graph): Operator {
    // const op = resolveOperator(node, opsets, this.opResolveRules);
    // op.initialize(node.attributes, node, graph);
    // return op;
    throw new Error('should not run into here');
  }
}

function safeInt32(num: number): number {
  return Math.min(2147483647, Math.max(num, -2147483648));
}
