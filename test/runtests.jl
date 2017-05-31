using Base.Test;

include("testPRcurve.jl");
include("testROCcurve.jl");

@testset "All" begin 
	@testset "PR curve" begin
		@testset "Partial PR curve" begin
			for i in 1:25000
				@test testPRcurvePartial();
			end
		end

		@testset "Full PR curve" begin
			for i in 1:25000
				@test testPRcurveFull();
			end
		end
	end
	@testset "ROC curve" begin
		@testset "Partial ROC curve" begin
			for i in 1:25000
				@test testROCcurvePartial();
			end
		end

		@testset "Full ROC curve" begin
			for i in 1:25000
				@test testROCcurveFull();
			end
		end
	end
end
