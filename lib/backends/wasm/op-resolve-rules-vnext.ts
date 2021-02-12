// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {OpSet} from '../../opset';

import {OperatorInfo} from './op-vnext';
import {BinaryOp} from './ops-vnext/binary-op';
import {Concat} from './ops-vnext/concat';
import {Conv} from './ops-vnext/conv';
import {ElementWise} from './ops-vnext/element-wise';
import {FusedConv} from './ops-vnext/fused-conv';
import {Gather} from './ops-vnext/gather';
import {Gemm} from './ops-vnext/gemm';
import {ConvNchwc} from './ops-vnext/hchwc-conv';
import {MatMul} from './ops-vnext/matmul';
import {Relu} from './ops-vnext/relu';
import {ReorderInput} from './ops-vnext/reorder_input';
import {ReorderOutput} from './ops-vnext/reorder_output';
import {Reshape} from './ops-vnext/reshape';
import {Resize} from './ops-vnext/resize';
import {Slice} from './ops-vnext/slice';
import {Unsqueeze} from './ops-vnext/unsqueeze';

export const OP_INFO_RESOLVE_RULES: ReadonlyArray<OpSet.ResolveRule<OperatorInfo>> = [
  ['Add', '', '7+', (node, opset) => new BinaryOp('Add', opset)],
  ['Concat', '', '4+', (node, opset) => new Concat(opset)],
  ['Conv', 'com.microsoft.nchwc', '1+', (node, opset) => new ConvNchwc(opset)],
  ['ReorderInput', 'com.microsoft.nchwc', '1+', (node, opset) => new ReorderInput(opset)],
  ['ReorderOutput', 'com.microsoft.nchwc', '1+', (node, opset) => new ReorderOutput(opset)],
  ['Conv', '', '1+', (node, opset) => new Conv(opset)],
  ['FusedConv', 'com.microsoft', '1+', (node, opset) => new FusedConv(opset)],
  ['Gather', '', '1+', (node, opset) => new Gather(opset)],
  ['Gemm', '', '7+', (node, opset) => new Gemm(opset)],
  ['Slice', '', '1+', (node, opset) => new Slice(opset)],
  ['MatMul', '', '1+', (node, opset) => new MatMul(opset)],
  ['Mul', '', '7+', (node, opset) => new BinaryOp('Mul', opset)],
  ['Relu', '', '1+', (node, opset) => new Relu(opset)],
  ['Reshape', '', '5+', (node, opset) => new Reshape(opset)],
  ['Resize', '', '10+', (node, opset) => new Resize(opset)],
  ['Sigmoid', '', '5+', (node, opset) => new ElementWise('Sigmoid', opset)],
  ['Tanh', '', '6+', (node, opset) => new ElementWise('Tanh', opset)],
  ['Unsqueeze', '', '1+', (node, opset) => new Unsqueeze(opset)],
  ['Upsample', '', '7+', (node, opset) => new Resize(opset, false)],
];
