import {onnx} from 'onnx-proto';

import {Attribute} from '../../../attribute';
import {OpSet} from '../../../opset';
import {OperatorInfo} from '../op-vnext';

export class ReorderOutput implements OperatorInfo {
  constructor(public opset: OpSet) {}

  inferenceType(inputTypes: readonly onnx.TensorProto.DataType[]): onnx.TensorProto.DataType[] {
    return [inputTypes[0]];
  }

  initializeAttributes(attribute: Attribute): void {
    attribute.set('channels', 'int', attribute.getInt('channels', 0));
    attribute.set('channels_last', 'int', attribute.getInt('channels_last', 0));
  }

  get hash() {
    return '';
  }
}
