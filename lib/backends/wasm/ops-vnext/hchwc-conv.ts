import {onnx} from 'onnx-proto';

import {Attribute} from '../../../attribute';
import {OpSet} from '../../../opset';
import {OperatorInfo} from '../op-vnext';

export class ConvNchwc implements OperatorInfo {
  constructor(public opset: OpSet) {}

  inferenceType(inputTypes: readonly onnx.TensorProto.DataType[]): onnx.TensorProto.DataType[] {
    return [inputTypes[0]];
  }

  initializeAttributes(attribute: Attribute): void {
    // if (attribute.contains('activation')) {
    //   attribute.set('activation', 'string', attribute.getString('activation', undefined));
    // }
    // if (attribute.contains('auto_pad')) {
    //   attribute.set('auto_pad', 'string', attribute.getString('auto_pad', undefined));
    // }
    // if (attribute.contains('dilations')) {
    //   attribute.set('dilations ', 'ints', attribute.getInts('dilations', undefined));
    // }
    // if (attribute.contains('group')) {
    //   attribute.set('group', 'int', attribute.getInt('group', 1));
    // }
    // if (attribute.contains('kernel_shape')) {
    //   attribute.set('kernel_shape', 'ints', attribute.getInts('kernel_shape', undefined));
    // }
    // if (attribute.contains('pads')) {
    //   attribute.set('pads', 'ints', attribute.getInts('pads', undefined));
    // }
    // if (attribute.contains('strides')) {
    //   attribute.set('strides', 'ints', attribute.getInts('strides', undefined));
    // }
  }

  get hash() {
    return '';
  }
}
