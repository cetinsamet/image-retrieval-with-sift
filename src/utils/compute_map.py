
import sys,pdb


def usage():
  sys.stderr.write("""usage: python compute_map.py resultfile.dat groundtruth.dat\n""")
  sys.exit(1)



def score_ap_from_ranks_1 (ranks, nres):
  """ Compute the average precision of one search.
  ranks = ordered list of ranks of true positives
  nres  = total number of positives in dataset  
  """
  
  # accumulate trapezoids in PR-plot
  ap=0.0

  # All have an x-size of:
  recall_step=1.0/nres
    
  for ntp,rank in enumerate(ranks):
      
    # y-size on left side of trapezoid:
    # ntp = nb of true positives so far
    # rank = nb of retrieved items so far
    if rank==0: precision_0=1.0
    else:       precision_0=ntp/float(rank)

    # y-size on right side of trapezoid:
    # ntp and rank are increased by one
    precision_1=(ntp+1)/float(rank+1)
    
    ap+=(precision_1+precision_0)*recall_step/2.0
        
  return ap

  
def get_groundtruth(gt_file):
  """ Read datafile holidays_images.dat and output a dictionary
  mapping queries to the set of positive results (plus a list of all
  images)"""
  gt={}
  with open(gt_file, "r") as gt_f:
    for line in gt_f:
      line = line.strip().split(" ")
      gt[line[0][:-1]] = set(line[1:])
  return gt

def parse_results(fname):
  """ go through the results file and return them in suitable
  structures"""
  for l in open(fname,"r"):
    fields=l.split()
    query_name=fields[0]
    ranks=[int(rank) for rank in fields[1::2]]
    yield (query_name,zip(ranks,fields[2::2]))


#########################################################################
# main program

if len(sys.argv)!=3: usage()

infilename=sys.argv[1]
gtfilename=sys.argv[2]

gt=get_groundtruth(gtfilename)

# sum of average precisions
sum_ap=0.
# nb of images so far
n=0

# loop over result lines
for query_name,results in parse_results(infilename):
  
  if query_name not in gt:
    print("unknown query ",query_name)
    sys.exit(1)

  # sort results by increasing rank
  results = list(results)
  results.sort()

  # ground truth
  gt_results=gt.pop(query_name)

  # ranks of true positives (not including the query)
  tp_ranks=[]

  # apply this shift to ignore null results
  rank_shift=0
  
  for rank,returned_name in results:      
    if returned_name==query_name:
      rank_shift=-1
    elif returned_name in gt_results:
      tp_ranks.append(rank+rank_shift)      

  sum_ap+=score_ap_from_ranks_1(tp_ranks,len(gt_results))
  n+=1


if gt:
  # some queries left
  print("no result for queries",gt.keys())
  sys.exit(1)

print("mAP for %s: %.5f"%(infilename,sum_ap/n))

