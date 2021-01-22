import {onnx} from 'onnx-proto';

import {Attribute} from '../../../attribute';
import {OpSet} from '../../../opset';
import {OperatorInfo} from '../op-vnext';

export class Slice implements OperatorInfo {
  constructor(public opset: OpSet) {}

  inferenceType(inputTypes: readonly onnx.TensorProto.DataType[]): onnx.TensorProto.DataType[] {
    return [inputTypes[0]];
  }

  initializeAttributes(attribute: Attribute): void {
    if (this.opset.version < 10) {
      attribute.set('axes', 'ints', attribute.getInts('axes', []));
    }
  }

  get hash() {
    return '';
  }
};
