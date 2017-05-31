using EduNetsEvaluate;
using MLBase;
using StatsBase;
using Plots;

function testROCcurvePartial()
	real = rand(0:1, 1024);
	predicted = rand(Float32, 1024);
	rocvec = MLBase.roc(real, predicted, nquantile(predicted, 100));
	TPRML = map(r->true_positive_rate(r), rocvec);
	FPRML = map(r->false_positive_rate(r), rocvec);
	real .+= 1;
	es = EvaluationState(predicted, real);
	roc = ROCcurve([es]);
	precisionResult = minimum((TPRML .- roc.TPR) .< 0.01);
	recallResult = minimum((FPRML .- roc.FPR) .< 0.01);

	#plotROCcurve(roc);
	#plot!(FPRML, TPRML, label = "MLBase ROC curve");
	return precisionResult && recallResult;
end

function testROCcurveFull()
	real1 = rand(0:1, 1024);
	predicted1 = rand(Float32, 1024);
	real2 = zeros(Int, 1024);
	predicted2 = rand(Float32, 1024);
	rocvec = MLBase.roc(vcat(real1, real2), vcat(predicted1, predicted2), nquantile(predicted1, 100));
	TPRML = map(r->true_positive_rate(r), rocvec);
	FPRML = map(r->false_positive_rate(r), rocvec);
	real1 .+= 1;
	real2 .+= 1;
	es1 = EvaluationState(predicted1, real1);
	es2 = EvaluationState(predicted2, real2);
	roc = ROCcurve([es1], [es2]);
	precisionResult = minimum((TPRML .- roc.TPR) .< 0.01);
	recallResult = minimum((FPRML .- roc.FPR) .< 0.01);

	#plotROCcurve(roc);
	#plot!(FPRML, TPRML, label = "MLBase ROC curve");
	return precisionResult && recallResult;
end
