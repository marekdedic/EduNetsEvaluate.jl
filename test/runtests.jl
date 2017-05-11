using Base.Test;

include("testPRcurve.jl");

@testset "All" begin 
	@testset "PR curve" begin
		@testset "Partial PR curve" begin
			for i in 1:10000
				@test testPRcurvePartial();
			end
		end

		@testset "Full PR curve" begin
			for i in 1:10000
				@test testPRcurveFull();
			end
		end
	end
end
