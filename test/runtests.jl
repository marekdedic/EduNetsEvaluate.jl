using Base.Test;

include("testPRcurve.jl");

@testset "PR curve" begin
	for i in 1:10000
		@test testPRcurve();
	end
end
