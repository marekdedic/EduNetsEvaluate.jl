import StatsBase;

export ROCcurveStage2

type ROCcurveStage2{A<:AbstractFloat}
	thresholds::Vector{A}
end

function ROCcurveStage2(S1::ROCcurveStage1; thresholdCount::Int = 100)::ROCcurveStage2
	return ROCcurveStage2(StatsBase.nquantile(S1.thresholds, thresholdCount));
end

