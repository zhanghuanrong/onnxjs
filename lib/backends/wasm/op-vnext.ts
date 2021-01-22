import {onnx} from 'onnx-proto';

import {Attribute} from '../../attribute';
import {OpSet} from '../../opset';

/**
 * This interface is used for WASM-vNEXT.
 */
export interface OperatorInfo {
  /**
   * Given types of each inputs, inference types of each outputs.
   */
  inferenceType(inputTypes: ReadonlyArray<onnx.TensorProto.DataType>): onnx.TensorProto.DataType[];

  /**
   * Given attributes from the graph, merge by default attributes value.
   */
  initializeAttributes(attribute: Attribute): void;

  /**
   * Get the HASH for the operator
   */
  readonly hash: string;

  /**
   * Current opset
   */
  readonly opset: OpSet;
}
