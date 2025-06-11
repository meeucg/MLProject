using System.Globalization;
using System.Runtime.CompilerServices;
using CsvHelper;
using CsvHelper.Configuration;

namespace MLProject.Services;

public static class CsvHelperDictionaryReader
{
    public static List<Dictionary<string, string>> ReadCsvAsDictionaries(
        string filePath, char delimiter = ',')
    {
        using var reader = new StreamReader(filePath);
        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            Delimiter = delimiter.ToString(),
            HasHeaderRecord = true
        };
        using var csv = new CsvReader(reader, config);

        csv.Read();
        csv.ReadHeader();

        var records = csv.GetRecords<dynamic>()
            .Cast<IDictionary<string, object>>()
            .Select(dict => dict.ToDictionary(
                kvp => kvp.Key,
                kvp => kvp.Value?.ToString() ?? string.Empty))
            .ToList();

        return records;
    }
    
    public static string GetCurrentFilePath([CallerFilePath] string path = null)
    {
        return path;
    }
}