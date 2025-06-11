using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using ProtoBuf;
using MathNet.Numerics;
using Newtonsoft.Json;

namespace MLProject.Services;

public class PCAServiceSvd
{
    private readonly int _dimensions;
    private Vector<double>? _meanVector;
    private Matrix<double>? _principalComponents;
    
    public List<double[]> TrainingData { get; set; }
    public List<double[]> CompressedData { get; private set; } = new();

    public PCAServiceSvd(List<double[]> trainingData, int dimensions)
    {
        Control.UseNativeMKL();
        TrainingData = trainingData;
        _dimensions = dimensions;
    }

    public void Learn()
    {
        var dataMatrix = DenseMatrix.Build.DenseOfRows(TrainingData);
        _meanVector = dataMatrix.ColumnSums() / dataMatrix.RowCount;

        var centeredMatrix = dataMatrix.Clone();
        for (int i = 0; i < centeredMatrix.RowCount; i++)
        {
            centeredMatrix.SetRow(i, centeredMatrix.Row(i) - _meanVector);
        }

        var svd = centeredMatrix.Svd(true);
        _principalComponents = svd.VT.SubMatrix(0, _dimensions, 0, svd.VT.ColumnCount);
    }

    public void CompressAll()
    {
        if (_meanVector == null || _principalComponents == null)
            throw new InvalidOperationException("Call Learn() before compression");

        var dataMatrix = DenseMatrix.Build.DenseOfRows(TrainingData);
        var centeredMatrix = dataMatrix.Clone();
        
        for (int i = 0; i < centeredMatrix.RowCount; i++)
        {
            centeredMatrix.SetRow(i, centeredMatrix.Row(i) - _meanVector);
        }

        var compressed = centeredMatrix * _principalComponents.Transpose();
        CompressedData = compressed.EnumerateRows().Select(r => r.ToArray()).ToList();
    }

    public double[] MapVector(double[] vector)
    {
        if (_meanVector == null || _principalComponents == null)
            throw new InvalidOperationException("Call Learn() before mapping vectors");

        var vectorRow = DenseVector.OfArray(vector) - _meanVector;
        var compressed = _principalComponents * vectorRow;
        
        return compressed.ToArray();
    }
    
    public void SaveToBinary()
    {
        if (_meanVector == null || _principalComponents == null)
            throw new InvalidOperationException("Call Learn() before saving");

        
        var filePath = Path.Combine("D:\\ModelTraining", 
            $"pca_model_{TrainingData[0].Length}To{_dimensions}.bin");

        var model = new PCAModelBinary
        {
            Mean = _meanVector.ToArray(),
            FlattenedPrincipalComponents = _principalComponents.ToColumnMajorArray(),
            PrincipalRows = _principalComponents.RowCount,
            PrincipalColumns = _principalComponents.ColumnCount
        };

        using (var file = File.Create(filePath))
        {
            Serializer.Serialize(file, model);
        }

        Console.WriteLine($"Model saved to {filePath}");
    }

    public void LearnFromBinary(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException("PCA model file not found", filePath);

        using (var file = File.OpenRead(filePath))
        {
            var model = Serializer.Deserialize<PCAModelBinary>(file);
            _meanVector = Vector<double>.Build.Dense(model.Mean);
            
            _principalComponents = Matrix<double>.Build.Dense(
                model.PrincipalRows,
                model.PrincipalColumns,
                model.FlattenedPrincipalComponents
            );
        }

        Console.WriteLine("Model loaded successfully from binary");
    }

    [ProtoContract]
    private class PCAModelBinary
    {
        [ProtoMember(1)]
        public double[] Mean { get; set; } = Array.Empty<double>();
    
        [ProtoMember(2)]
        public double[] FlattenedPrincipalComponents { get; set; } = Array.Empty<double>();
    
        [ProtoMember(3)]
        public int PrincipalRows { get; set; }
    
        [ProtoMember(4)]
        public int PrincipalColumns { get; set; }
    }
}