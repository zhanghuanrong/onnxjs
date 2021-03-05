import sys
import re

def ProcessPrintOutFile(file_path):
  with open(file_path, encoding='utf8') as infile:
    run_latencies = []
    sum_run_latencies = []
    op_names = {}
    node_names = {}
    op_latencies = {}
    run = 0
    max_op = -1
    sum_op_time = 0
    for line in infile:
      #'End Run() from [thread:58628] at 365214 total Run() latency:9804 us'
      match = re.search('End Run.+latency:(?P<RunLatency>[0-9]+) us', line)
      if match:
        run_latency = match.group('RunLatency')
        if run == 0:
          print('First Run:', run_latency, 'us, sum all ops:', sum_op_time, 'us')
        elif run >= 3:
          run_latencies.append(int(run_latency))
          sum_run_latencies.append(sum_op_time)
        run += 1
        sum_op_time = 0
        continue

      # Kernel 0, name:fused Conv_15, op_type:FusedConv, Start:1540954, End:1541334, latency:380 us'
      matching_node_name = True
      match = re.search('Kernel (?P<idx>[0-9]+), name:(?P<name>[^,]+), op_type:(?P<op>[^,]+), .*, latency:(?P<OpLatency>[0-9]+) us', line)
      if not match:
        # for old version of output
        # Kernel 0, op_name:Conv, Start:355410, End:355860, latency:450 us'
        match = re.search(' Kernel (?P<idx>[0-9]+), op_name:(?P<op>[^,]+), .*, latency:(?P<OpLatency>[0-9]+) us', line)
        matching_node_name = False
      if match:
        sum_op_time += int(match.group('OpLatency'))
        if run == 0:
          idx = int(match.group('idx'))
          if max_op < idx:
            max_op = idx
          op_names.update({idx : match.group('op')})
          if matching_node_name:
            node_names.update({idx: match.group('name')})
          else:
            node_names.update({idx: ''})
        elif run == 3:
          op_latencies.update({int(match.group('idx')) : [int(match.group('OpLatency'))]})
        elif run > 3:
          op_latencies[int(match.group('idx'))].append(int(match.group('OpLatency')))

    # avg latency in first run
    if len(run_latencies) > 0:
      avg_run = round(sum(run_latencies) / len(run_latencies))
      print('Average Inference Latency:', avg_run, "us, in", len(run_latencies), "runs after first run")
    if len(sum_run_latencies) > 0:
      avg_sum_ops = round(sum(sum_run_latencies) / len(sum_run_latencies))
      print('Average Inference Latency by sum ops:', avg_sum_ops, "us, in", len(sum_run_latencies), "runs after first run")

    # do op name aggregation
    agg_latencies = {}
    for idx in range(max_op + 1):
      assert(len(op_latencies[idx]) == len(run_latencies))
      avg_op_latency = sum(op_latencies[idx]) / len(run_latencies)
      if op_names[idx] not in agg_latencies:
        agg_latencies[op_names[idx]] = [avg_op_latency]
      else:
        agg_latencies[op_names[idx]].append(avg_op_latency)

    # print aggregation latency by op_name (sum, #)
    print('---------average cross runs for lentency aggregated by op_type in one run------------')
    sumed_agg = [ (pair[0], sum(pair[1]), len(pair[1])) for pair in list(agg_latencies.items())]
    ss = sorted(sumed_agg, key=lambda x: x[1], reverse=True)
    for agg in ss:
      print("  {}: {} us, {} operator(s)".format(agg[0], str(round(agg[1])), str(agg[2])))

    # aggregate fused depthwise conv in tfjs teams model
    print('---------aggregate fused depthwise conv in tfjs teams model---------------------------------------')
    fusedDepthWiseConvs = [
        'fused Conv_33', 'fused Conv_56', 'fused Conv_79', 'fused Conv_103', 'fused Conv_126',
        'fused Conv_150', 'fused Conv_174', 'fused Conv_197', 'fused Conv_221', 'fused Conv_245',
        'fused Conv_269', 'fused Conv_292', 'fused Conv_316']
    sum_fusedDepthwiseConv_latency = 0
    op_count_fusedDepthwiseConv = 0
    for idx in range(max_op + 1):
      if node_names[idx] in fusedDepthWiseConvs:
        sum_fusedDepthwiseConv_latency += round(sum(op_latencies[idx]) / len(run_latencies))
        op_count_fusedDepthwiseConv += 1
    print('  depthwise conv total latency in single inference is {} us for {} ops'.format(sum_fusedDepthwiseConv_latency, op_count_fusedDepthwiseConv))

    # aggregate normal 3x3 conv in tfjs teams model
    print('---------aggregate normal 3x3 conv (non depth wise, non point wise) in tfjs teams model---------------------------------------')
    normalConvs = [
        'fused Conv_15', 'fused Conv_349', 'fused Conv_367', 'fused Conv_395', 'fused Conv_413']
    sum_NormalConv_latency = 0
    op_count_normalConv = 0
    for idx in range(max_op + 1):
      if node_names[idx] in normalConvs:
        sum_NormalConv_latency += round(sum(op_latencies[idx]) / len(run_latencies))
        op_count_normalConv += 1
    print('  normal 3x3 conv total latency in single inference is {} us for {} ops'.format(sum_NormalConv_latency, op_count_normalConv))


    print('---------Top 200 heaviest operators---------------------------------------')
    ss = sorted([(idx, round(sum(op_latencies[idx]) / len(run_latencies))) for idx in range(max_op+1)], key=lambda x:x[1], reverse=True)
    for sidx in range(min(len(ss), 200)):
      idx = ss[sidx][0]
      print('  {} -- {} ({}), {} us'.format(idx, node_names[idx], op_names[idx], ss[sidx][1]))

    print('---------average latency for each op cross runs---------------------------------------')
    for idx in range(max_op + 1):
      assert(len(op_latencies[idx]) == len(run_latencies))
      avg_op_latency = round(sum(op_latencies[idx]) / len(run_latencies))
      print('  {} -- {} ({}), {} us'.format(idx, node_names[idx], op_names[idx], avg_op_latency))


if __name__ == "__main__":
  ProcessPrintOutFile(sys.argv[1])

