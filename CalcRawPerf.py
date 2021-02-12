import sys
import re

def ProcessPrintOutFile(file_path):
  with open(file_path, encoding='utf8') as infile:
    run_latencies = []
    op_names = {}
    op_latencies = {}
    run = 0
    max_op = -1
    for line in infile:
      #'End Run() from [thread:58628] at 365214 total Run() latency:9804 us'
      match = re.search('End Run.+latency:(?P<RunLatency>[0-9]+) us', line)
      if match:
        run_latency = match.group('RunLatency')
        if run == 0:
          print('First Run:', run_latency, 'us')
        else:
          run_latencies.append(int(run_latency))
        run += 1
        continue

      # Kernel 0, op_name:Conv, Start:355410, End:355860, latency:450 us'
      match = re.search(' Kernel (?P<idx>[0-9]+), op_name:(?P<op>[^,]+), .*, latency:(?P<OpLatency>[0-9]+) us', line)
      if match:
        if run == 0:
          idx = int(match.group('idx'))
          if max_op < idx:
            max_op = idx
          op_names.update({idx : match.group('op')})
        elif run == 1:
          op_latencies.update({int(match.group('idx')) : [int(match.group('OpLatency'))]})
        else:
          op_latencies[int(match.group('idx'))].append(int(match.group('OpLatency')))

    # avg latency in first run
    avg_run = round(sum(run_latencies) / len(run_latencies))
    print('Average Inference Latency:', avg_run, "us, in", len(run_latencies), "runs after first run")

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
    print('---------average cross runs for lentency aggregated by name in one run------------')
    sumed_agg = [ (pair[0], sum(pair[1]), len(pair[1])) for pair in list(agg_latencies.items())]
    ss = sorted(sumed_agg, key=lambda x: x[1], reverse=True)
    for agg in ss:
      print("  {}: {} us, {} operator(s)".format(agg[0], str(round(agg[1])), str(agg[2])))

    print('---------average latency for each op cross runs---------------------------------------')
    for idx in range(max_op + 1):
      assert(len(op_latencies[idx]) == len(run_latencies))
      avg_op_latency = round(sum(op_latencies[idx]) / len(run_latencies))
      print('  {} -- {}, {} us'.format(idx, op_names[idx], avg_op_latency))

    print('---------Top 10 heaviest operators---------------------------------------')
    ss = sorted([(idx, round(sum(op_latencies[idx]) / len(run_latencies))) for idx in range(max_op+1)], key=lambda x:x[1], reverse=True)
    for sidx in range(min(len(ss), 10)):
      idx = ss[sidx][0]
      print('  {} -- {}, {} us'.format(idx, op_names[idx], ss[sidx][1]))


if __name__ == "__main__":
  ProcessPrintOutFile(sys.argv[1])

