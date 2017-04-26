import StatsBase;

export PRcurveStage2

type PRcurveStage2{A<:AbstractFloat}
	thresholds::Vector{A}
end

function PRcurveStage2(S1::PRcurveStage1; thresholdCount::Int = 100)::PRcurveStage2
	return PRcurveStage2(StatsBase.nquantile(S1.thresholds, thresholdCount));
end

