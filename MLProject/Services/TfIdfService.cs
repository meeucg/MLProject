using Accord.MachineLearning;

namespace MLProject.Services;

public class TfIdfService(List<string> trainingData)
{
    public readonly List<string> TrainingData = trainingData;
    public List<double[]> CompressedTrainingData { get; } = [];
    private string[][] _tokens = [];

    private readonly TFIDF _codebook = new()
    {
        Tf = TermFrequency.Log,
        Idf = InverseDocumentFrequency.Default
    };


    public void Learn()
    {
        _tokens = TrainingData.ToArray().Tokenize();
        _codebook.Learn(_tokens);
    }

    public void CompressAllTrainingData()
    {
        foreach (var document in _tokens!)
        {
            var compressed = _codebook.Transform(document);
            CompressedTrainingData.Add(compressed);
        }
    }

    public double[] CompressDocument(string document)
    {
        TrainingData.Add(document);
        var tokenizedDoc = document.Tokenize();
        var compressedDoc = _codebook.Transform(tokenizedDoc);
        CompressedTrainingData.Add(compressedDoc);
        return compressedDoc;
    }
}

