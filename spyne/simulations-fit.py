#!/usr/bin/env python
import test, model, sys
import numpy as np
import neural

#print __name__
if __name__ == "__main__":
    num = int(sys.argv[1])
    print "Starting, N=%d" % num
    fout=file("fit-simulations.txt", "w")
    fails={}
    for i in xrange(num):
        for x in [x/100.0 for x in xrange(100, 325, 50)]:
            if x not in fails.keys():
                fails[x]=0
            m=model.Model3()
            sn = m.GetGroupByName("SN")
            sp = m.GetGroupByName("SP")
            sn.SetActivationFunction(np.frompyfunc(lambda x: neural.STanh_plus(x, gain=x), 1,1))
            sp.SetActivationFunction(np.frompyfunc(lambda x: neural.STanh_plus(x, gain=x), 1,1))
            t=test.Test1()
            t.model=m
            print "[%03d] g=%.1f, model=%s" % (i, x, t.model.name)
            res    = {}
            while len(res.keys()) != 3:
                res=t.doTest()
                if len(res.keys()) != 3:
                    fails[x]+=1
            t.task.Stats()
            line = "%s\t" * 4
            line = line[:-1]+"\n"
            fout.write(line % (x, res['AB'], res['CD'], res['EF']))
            fout.flush()
        print fails
    fout.close()
    print fails
