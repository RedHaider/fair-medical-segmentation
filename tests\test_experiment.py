from fair_medseg.experiment import data,dice,evaluate
def test_dice_perfect(): assert abs(dice(__import__('numpy').array([1,0]),__import__('numpy').array([1,0]))-1)<1e-6
def test_report(): assert all(0<=v<=1 for m in evaluate(3).values() for v in m.values())

