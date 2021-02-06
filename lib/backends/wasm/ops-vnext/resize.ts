import {onnx} from 'onnx-proto';

import {Attribute} from '../../../attribute';
import {OpSet} from '../../../opset';
import {OperatorInfo} from '../op-vnext';

export class Resize implements OperatorInfo {
  constructor(public opset: OpSet, public isResize = true) {}

  inferenceType(inputTypes: readonly onnx.TensorProto.DataType[]): onnx.TensorProto.DataType[] {
    return [inputTypes[0]];
  }

  initializeAttributes(attribute: Attribute): void {
    attribute.set('mode', 'string', attribute.getString('mode', 'nearest'));

    if (this.opset.version >= 11) {
      attribute.set(
          'coordinate_transformation_mode', 'string',
          attribute.getString('coordinate_transformation_mode', 'half_pixel'));
      attribute.set('cubic_coeff_a', 'float', attribute.getFloat('cubic_coeff_a', -0.75));
      attribute.set('exclude_outside', 'int', attribute.getInt('exclude_outside', 0));
      attribute.set('extrapolation_value', 'float', attribute.getFloat('extrapolation_value', 0));
      attribute.set('nearest_mode', 'string', attribute.getString('nearest_mode', 'round_prefer_floor'));
    }
  }

  get hash() {
    return '';
  }
}
