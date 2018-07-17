import sys
import os

def main(argv):
    assert len(argv) == 2, "Usage: python convert_for_eval.py result.out"
    dir = os.path.dirname(argv[1])
    result = os.path.basename(argv[1])

    with open(os.path.join(dir, "converted_{}".format(result)), "w") as out:
        with open(os.path.join(dir, result), "r") as inp:
            for line in inp:
                query, results = line.strip().split(sep=':')
                results = results[1:].split(sep=' ')
                pairs = []
                for p in zip(results[0::2], results[1::2]):
                    pairs.append(p)
                # Sort pairs
                pairs.sort(key=lambda x: float(x[0]))
                result_line = "{}".format(query)
                i = 0
                for pair in pairs:
                    result_line = "{} {} {}".format(result_line, i, pair[1])
                    i += 1
                result_line += '\n'
                out.write(result_line)

if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv)



