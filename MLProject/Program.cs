using MLProject.Services;
using Distance = MathNet.Numerics.Distance;

// Usage:
var currentFilePath = CsvHelperDictionaryReader.GetCurrentFilePath();
var contentPath = Path.Combine(Path.GetDirectoryName(currentFilePath)!, "Content/BooksDatasetClean.csv");

var database = CsvHelperDictionaryReader.ReadCsvAsDictionaries(Path.GetFullPath(contentPath));
var texts = new List<string>();
int count = 0;

for (var index = 0; index < 7000; index++)
{
    var book = database[index];
    if (book["Description"] != "")
    {
        texts.Add($"{book["Description"]}");
        count++;
    }
}
var text1 = "Set in the dystopian superstate of Oceania, *1984* follows Winston Smith, a disillusioned Party member who secretly rebels against the all-seeing regime of Big Brother. Rewriting historical records for the Ministry of Truth, Winston grapples with the Party’s manipulation of reality through Newspeak and \"doublethink\". His forbidden affair with Julia and alliance with the enigmatic O’Brien—who later betrays him—culminate in a harrowing journey of torture and psychological reprogramming in Room 101, where even love becomes a weapon of control. Orwell’s chilling critique of totalitarianism explores the erasure of truth, individuality, and the human spirit under omnipresent surveillance";
var text2 = "A haunting prophecy of authoritarianism, 1984 paints a world where the Party enforces loyalty through telescreens, Thought Police, and the relentless mantra: War is Peace. Freedom is Slavery. Ignorance is Strength. Winston’s quiet defiance—documenting his dissent in a forbidden diary and pursuing a clandestine romance—collapses when he discovers the Brotherhood’s resistance is a trap. Broken by betrayal and forced to confront his deepest fears, he surrenders to the Party’s doctrine, epitomizing the novel’s warning: power lies in crushing the mind’s capacity to resist. Decades after publication, its concepts like \"Big Brother\" and \"Orwellian\" remain stark symbols of dystopian reality.";
texts.Add(text1);
texts.Add(text2);

Console.WriteLine(count); //Count of books in training 

var tfIdfService = new TfIdfService(texts);
tfIdfService.Learn();
tfIdfService.CompressAllTrainingData();

Console.WriteLine(tfIdfService.CompressedTrainingData[0].Length); //TfIdf dimensions

var pcaService = new PCAServiceSvd(tfIdfService.CompressedTrainingData.ToList(), 10000);

// //Compute PCA from training data in runtime
//  pcaService.Learn(); 

//Load from .bin training data file
pcaService.TrainingData = tfIdfService.CompressedTrainingData;
pcaService.LearnFromBinary("D:\\ModelTraining\\pca_model_32946To10000.bin");

pcaService.CompressAll();

var newText = "Nineteen Eighty-four, novel by English author George Orwell published in 1949 as a warning against totalitarianism. The novel’s chilling dystopia made a deep impression on readers, and Orwell’s ideas entered mainstream culture in a way achieved by very few books. The book’s title and many of its concepts, such as Big Brother and the Thought Police, are instantly recognized and understood, often as bywords for modern social and political abuses.";
var newTfIdf = tfIdfService.CompressDocument(newText);
var newVector = pcaService.MapVector(newTfIdf);

Console.WriteLine("\n" + newText + "\n");
var sims = new List<(double sim, int id)>();
for (int i = 0; i < pcaService.CompressedData.Count; i++)
{
    var similarity = double.Abs(1 - Distance.Cosine(newVector, pcaService.CompressedData[i]));
    sims.Add((similarity, i));
}
sims.Sort((s1, s2) => s1.sim.CompareTo(s2.sim));
foreach (var sim in sims.Skip(sims.Count-10).Take(10))
{
    Console.WriteLine(texts[sim.id] + "\t\t" + sim.sim + "\n");
}