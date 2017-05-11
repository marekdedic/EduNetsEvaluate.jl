using EduNetsEvaluate;
using MLBase;
using StatsBase;
using Plots;

function testPRcurvePartial()
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

function testPRcurveFull()
	real1 = rand(0:1, 1024);
	predicted1 = rand(Float32, 1024);
	real2 = zeros(Int, 1024);
	predicted2 = rand(Float32, 1024);
	rocvec = roc(vcat(real1, real2), vcat(predicted1, predicted2), nquantile(predicted1, 100));
	precisionML = map(r->precision(r), rocvec);
	recallML = map(r->recall(r), rocvec);
	real1 .+= 1;
	real2 .+= 1;
	es1 = EvaluationState(predicted1, real1);
	es2 = EvaluationState(predicted2, real2);
	pr = PRcurve([es1], [es2]);
	precisionResult = minimum((precisionML .- pr.precision) .< 0.01);
	recallResult = minimum((recallML .- pr.recall) .< 0.01);

	#plotPRcurve(pr);
	#plot!(recallML, precisionML, label = "MLBase PR curve");
	return precisionResult && recallResult;
end
