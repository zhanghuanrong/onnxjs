import {onnx} from 'onnx-proto';

import {Attribute} from '../../../attribute';
import {OpSet} from '../../../opset';
import {OperatorInfo} from '../op-vnext';

export class Gemm implements OperatorInfo {
  constructor(public opset: OpSet) {}

  inferenceType(inputTypes: readonly onnx.TensorProto.DataType[]): onnx.TensorProto.DataType[] {
    return [inputTypes[0]];
  }

  initializeAttributes(attribute: Attribute): void {
    attribute.set('transA', 'int', attribute.getInt('transA', 0));
    attribute.set('transB', 'int', attribute.getInt('transB', 0));
    attribute.set('alpha', 'float', attribute.getFloat('alpha', 1.0));
    attribute.set('beta', 'float', attribute.getFloat('beta', 1.0));
  }

  get hash() {
    return '';
  }
}
