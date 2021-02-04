import {onnx} from 'onnx-proto';

import {Attribute} from '../../../attribute';
import {OpSet} from '../../../opset';
import {OperatorInfo} from '../op-vnext';

export class BinaryOp implements OperatorInfo {
  constructor(protected opType: string, public opset: OpSet, protected isBooleanOutput = false) {}

  inferenceType(inputTypes: readonly onnx.TensorProto.DataType[]): onnx.TensorProto.DataType[] {
    return [this.isBooleanOutput ? onnx.TensorProto.DataType.BOOL : inputTypes[0]];
  }

  initializeAttributes(attribute: Attribute): void {}

  get hash() {
    return '';
  }
}
