module EduNetsEvaluate

include("EvaluationState.jl");

include("PRcurve/PRcurveStage1.jl");
include("PRcurve/PRcurveStage2.jl");
include("PRcurve/PRcurveStage3.jl");
include("PRcurve/PRcurveStage4.jl");
include("PRcurve/PRcurveStage5.jl");
include("PRcurve/PRcurve.jl");

include("ROCcurve/ROCcurveStage1.jl");
include("ROCcurve/ROCcurveStage2.jl");
include("ROCcurve/ROCcurveStage3.jl");
include("ROCcurve/ROCcurveStage4.jl");
include("ROCcurve/ROCcurveStage5.jl");
include("ROCcurve/ROCcurve.jl");

end
