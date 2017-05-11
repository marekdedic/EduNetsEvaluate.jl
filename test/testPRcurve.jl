using EduNetsEvaluate;
using MLBase;
using StatsBase;
using Plots;

function testPRcurve()
	real = rand(0:1, 1024);
	predicted = rand(Float32, 1024);
	rocvec = roc(real, predicted, nquantile(predicted, 100));
	precisionML = map(r->precision(r), rocvec);
	recallML = map(r->recall(r), rocvec);
	real .+= 1;
	es = EvaluationState(predicted, real);
	pr = PRcurve([es]);
	precisionResult = minimum((precisionML .- pr.precision) .< 0.01);
	recallResult = minimum((recallML .- pr.recall) .< 0.01);

	#plotPRcurve(pr);
	#plot!(recallML, precisionML, label = "MLBase PR curve");
	return precisionResult && recallResult;
end
