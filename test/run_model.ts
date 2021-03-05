import {readdirSync, readFileSync} from 'fs';
import {onnx as onnxProto} from 'onnx-proto';
import {extname, join} from 'path';
import {inspect} from 'util';

import * as api from '../lib/api';
import {fromInternalTensor} from '../lib/api/tensor-impl-utils';
import {Logger} from '../lib/instrument';
import {Tensor} from '../lib/tensor';


export interface NamedTensor extends api.Tensor {
  name: string;
}


async function initializeSession(modelFilePath: string, backendHint: string) {
  const profilerConfig = undefined;
  const sessionConfig: api.InferenceSession.Config = {backendHint, profiler: profilerConfig};
  const session = new api.InferenceSession(sessionConfig);

  try {
    await session.loadModel(modelFilePath);
  } catch (e) {
    Logger.error('TestRunner', `Failed to load model from file: ${modelFilePath}. Error: ${inspect(e)}`);
    throw e;
  }

  Logger.verbose('TestRunner', `Finished loading model from file: ${modelFilePath}`);

  return session;
}


async function loadTensors(
    dataDir: string, dataFiles: ReadonlyArray<string>, inputs: NamedTensor[], outputs: NamedTensor[]) {
  let dataFileType: 'none'|'pb'|'npy' = 'none';

  for (const dataFile of dataFiles) {
    const ext = extname(dataFile);
    if (ext.toLowerCase() === '.pb' || ext.toLowerCase() === '.tpb') {
      if (dataFileType === 'none') {
        dataFileType = 'pb';
      }
      if (dataFileType !== 'pb') {
        throw new Error(`cannot load data other than pb format`);
      }

      const tensor_file = join(dataDir, dataFile);
      const buf = readFileSync(tensor_file);
      const tensorProto = onnxProto.TensorProto.decode(Buffer.from(buf));
      const tensor = Tensor.fromProto(tensorProto);
      // add property 'name' to the tensor object.
      const namedTensor = fromInternalTensor(tensor) as unknown as NamedTensor;
      namedTensor.name = tensorProto.name;

      const dataFileBasename = dataFile.split(/[/\\]/).pop()!;
      if (dataFileBasename.indexOf('input') !== -1) {
        inputs.push(namedTensor);
      } else if (dataFileBasename.indexOf('output') !== -1) {
        outputs.push(namedTensor);
      }
    } else {
      continue;
    }
  }
}


export async function runModel(modelUrl: string, dataDir: string): Promise<void> {
  const session = await initializeSession(modelUrl, 'wasm');
  const dataFiles = readdirSync(dataDir);
  let inputs: NamedTensor[] = [];
  let outputs: NamedTensor[] = [];
  loadTensors(dataDir, dataFiles, inputs, outputs);

  try {
    await session.run(inputs!);

    await session.run(inputs!);
    await session.run(inputs!);

    for (let kadsf = 1; kadsf <= 50; kadsf++) {
      await session.run(inputs!);
    }

  } catch (e) {
    Logger.error('Run Model', `  Result: FAILED`);
    throw e;
  }
}

runModel('test/data/teamsmodel_o2/o2.onnx', 'test/data/teamsmodel_o2/test_dataset_0');
