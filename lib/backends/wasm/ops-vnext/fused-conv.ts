import {onnx} from 'onnx-proto';

import {Attribute} from '../../../attribute';
import {OpSet} from '../../../opset';
import {OperatorInfo} from '../op-vnext';

export class FusedConv implements OperatorInfo {
  constructor(public opset: OpSet) {}

  inferenceType(inputTypes: readonly onnx.TensorProto.DataType[]): onnx.TensorProto.DataType[] {
    return [inputTypes[0]];
  }

  initializeAttributes(attribute: Attribute): void {
    attribute.set('activation', 'string', attribute.getString('activation', 'Relu'));
    attribute.set('auto_pad', 'string', attribute.getString('auto_pad', 'NOTSET'));
    // attribute.set('dilations ', 'ints', attribute.getInts('dilations', 1));
    attribute.set('group', 'int', attribute.getInt('group', 1));
    // attribute.set('kernel_shape', 'ints', attribute.getFloat('kernel_shape', 0));
    // attribute.set('pads', 'string', attribute.getString('nearest_mode', 'round_prefer_floor'));
  }

  get hash() {
    return '';
  }
}
