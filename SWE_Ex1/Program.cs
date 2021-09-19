using Microsoft.ML;
using System;

namespace SWE_Ex1
{
    class SampleMNISTData
    {
        internal static readonly InputData MNIST1 = new InputData()
        {
            PixelValues = new float[] { 0, 0, 0, 0, 14, 13, 1, 0, 0, 0, 0, 5, 16, 16, 2, 0, 0, 0, 0, 14, 16, 12, 0, 0, 0, 1, 10, 16, 16, 12, 0, 0, 0, 3, 12, 14, 16, 9, 0, 0, 0, 0, 0, 5, 16, 15, 0, 0, 0, 0, 0, 4, 16, 14, 0, 0, 0, 0, 0, 1, 13, 16, 1, 0 }
        }; //num 1
}

    class Program
    {
        static void Main(string[] args)
        {
            // STEP 1: Common data loading configuration
            var trainData = mlContext.Data.LoadFromTextFile(path: TrainDataPath,
                                    columns: new[]
                                    {
                            new TextLoader.Column(nameof(InputData.PixelValues), DataKind.Single, 0, 63),
                            new TextLoader.Column("Number", DataKind.Single, 64)
                                    },
                                    hasHeader: false,
                                    separatorChar: ','
                                    );


            var testData = mlContext.Data.LoadFromTextFile(path: TestDataPath,
                                    columns: new[]
                                    {
                            new TextLoader.Column(nameof(InputData.PixelValues), DataKind.Single, 0, 63),
                            new TextLoader.Column("Number", DataKind.Single, 64)
                                    },
                                    hasHeader: false,
                                    separatorChar: ','
                                    );

            // STEP 2: Common data process configuration with pipeline data transformations
            // Use in-memory cache for small/medium datasets to lower training time. Do NOT use it (remove .AppendCacheCheckpoint()) when handling very large datasets.
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Number").
                                Append(mlContext.Transforms.Concatenate("Features", nameof(InputData.PixelValues)).AppendCacheCheckpoint(mlContext));

            // STEP 3: Set the training algorithm, then create and config the modelBuilder
            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer).Append(mlContext.Transforms.Conversion.MapKeyToValue("Number", "Label"));

            // STEP 4: Train the model fitting to the DataSet            
            ITransformer trainedModel = trainingPipeline.Fit(trainData);


            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Number", scoreColumnName: "Score");

            Common.ConsoleHelper.PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);



            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<InputData, OutPutData>(trainedModel);

            var resultprediction1 = predEngine.Predict(SampleMNISTData.MNIST1);

            Console.WriteLine($"Actual: 7     Predicted probability:       zero:  {resultprediction1.Score[0]:0.####}");
            Console.WriteLine($"                                           One :  {resultprediction1.Score[1]:0.####}");
            Console.WriteLine($"                                           two:   {resultprediction1.Score[2]:0.####}");
            Console.WriteLine($"                                           three: {resultprediction1.Score[3]:0.####}");
            Console.WriteLine($"                                           four:  {resultprediction1.Score[4]:0.####}");
            Console.WriteLine($"                                           five:  {resultprediction1.Score[5]:0.####}");
            Console.WriteLine($"                                           six:   {resultprediction1.Score[6]:0.####}");
            Console.WriteLine($"                                           seven: {resultprediction1.Score[7]:0.####}");
            Console.WriteLine($"                                           eight: {resultprediction1.Score[8]:0.####}");
            Console.WriteLine($"                                           nine:  {resultprediction1.Score[9]:0.####}");
            Console.WriteLine();
        }
    }
}
