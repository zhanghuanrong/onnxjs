import {onnx} from 'onnx-proto';

import {Attribute} from '../../../attribute';
import {OpSet} from '../../../opset';
import {OperatorInfo} from '../op-vnext';

export class ElementWise implements OperatorInfo {
  constructor(protected opType: string, public opset: OpSet) {}

  inferenceType(inputTypes: readonly onnx.TensorProto.DataType[]): onnx.TensorProto.DataType[] {
    return [inputTypes[0]];
  }

  initializeAttributes(attribute: Attribute): void {}

  get hash() {
    return '';
  }
}
