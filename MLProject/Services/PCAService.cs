using Accord.Statistics.Analysis;
using Accord.Statistics.Models.Regression.Linear;
using MathNet.Numerics.LinearAlgebra.Double;

namespace MLProject.Services;

public class PCAService(List<double[]> trainingData, int dimensions)
{
    public List<double[]> TrainingData { get; } = trainingData;
    public List<double[]> CompressedData { get; set; } = [];

    public MultivariateLinearRegression? LearnedData { get; set; }
    
    private readonly PrincipalComponentAnalysis _pca = new()
    {
        Method = PrincipalComponentMethod.Center,
        Whiten = false,
        NumberOfOutputs = dimensions
    };

    public void Learn() 
    {
        var dataMatrix = DenseMatrix.Build.DenseOfRows(TrainingData);
        var meanVector = dataMatrix.ColumnSums() / dataMatrix.RowCount;

        var centeredMatrix = dataMatrix.Clone();
        for (var i = 0; i < centeredMatrix.RowCount; i++)
        {
            centeredMatrix.SetRow(i, centeredMatrix.Row(i) - meanVector);
        }
        LearnedData = _pca.Learn(
            centeredMatrix.EnumerateRows()
            .Select(row => row.ToArray())
            .ToArray()
            );
    }

    public void CompressAll()
    {
        CompressedData = _pca.Transform(TrainingData.ToArray()).ToList();
    }

    public double[] MapVector(double[] vector)
    {
        TrainingData.Add(vector);
        var compressed = _pca.Transform(vector);
        CompressedData.Add(compressed);
        return compressed;
    }
}
