/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.basicdataset;

import ai.djl.basicdataset.nlp.StanfordQuestionAnsweringDataset;
import ai.djl.basicdataset.utils.TextData;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

@SuppressWarnings("unchecked")
public class StanfordQuestionAnsweringDatasetTest {

    private static final int EMBEDDING_SIZE = 15;

    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/47
    @Test
    public void testPrepare1() throws TranslateException, IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            StanfordQuestionAnsweringDataset stanfordQuestionAnsweringDataset =
                    StanfordQuestionAnsweringDataset.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optLimit(10)
                            .optUsage(Dataset.Usage.TEST)
                            .build();

            stanfordQuestionAnsweringDataset.prepare();
            Assert.assertEquals(stanfordQuestionAnsweringDataset.size(), 10);
        }
    }

    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/47
    @Test
    public void testPrepare2() throws TranslateException, IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            StanfordQuestionAnsweringDataset stanfordQuestionAnsweringDataset =
                    StanfordQuestionAnsweringDataset.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optUsage(Dataset.Usage.TEST)
                            .build();

            stanfordQuestionAnsweringDataset.prepare();
            Assert.assertEquals(stanfordQuestionAnsweringDataset.size(), 11873);
        }
    }

    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/47
    @Test
    public void testGet1() throws TranslateException, IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            StanfordQuestionAnsweringDataset stanfordQuestionAnsweringDataset =
                    StanfordQuestionAnsweringDataset.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optLimit(10)
                            .optUsage(Dataset.Usage.TEST)
                            .build();

            stanfordQuestionAnsweringDataset.prepare();
            // normal get
            Record record = stanfordQuestionAnsweringDataset.get(manager, 9);
            Assert.assertEquals(record.getData().get("title").getShape().get(0), 1);
            Assert.assertEquals(record.getData().get("question").getShape().get(0), 10);
            Assert.assertEquals(record.getLabels().size(), 3);
        }
    }

    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/47
    @Test
    public void testGet2() throws TranslateException, IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            StanfordQuestionAnsweringDataset stanfordQuestionAnsweringDataset =
                    StanfordQuestionAnsweringDataset.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optLimit(10)
                            .optUsage(Dataset.Usage.TEST)
                            .build();

            stanfordQuestionAnsweringDataset.prepare();
            // out of preprocessed bound to get
            Record record = stanfordQuestionAnsweringDataset.get(manager, 20);
            Assert.fail("Should fail at out-of-bound get!");
        } catch (IndexOutOfBoundsException exception) {
            Assert.assertTrue(exception.getMessage().contains("Index: 20, Size: 13"));
        }
    }

    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/47
    @Test
    public void testGetData1() throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            StanfordQuestionAnsweringDataset stanfordQuestionAnsweringDataset =
                    StanfordQuestionAnsweringDataset.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optLimit(350)
                            .optUsage(Dataset.Usage.TEST)
                            .build();

            Map<String, Object> data =
                    (Map<String, Object>) stanfordQuestionAnsweringDataset.getData();
            Assert.assertEquals(((List<Object>) data.get("data")).size(), 35);
        }
    }

    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/47
    @Test
    public void testGetData2() throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            StanfordQuestionAnsweringDataset stanfordQuestionAnsweringDataset =
                    StanfordQuestionAnsweringDataset.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optLimit(350)
                            .optUsage(Dataset.Usage.TRAIN)
                            .build();

            Map<String, Object> data =
                    (Map<String, Object>) stanfordQuestionAnsweringDataset.getData();
            Assert.assertEquals(((List<Object>) data.get("data")).size(), 442);
        }
    }

    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/47
    @Test
    public void testScenario1() throws TranslateException, IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            StanfordQuestionAnsweringDataset stanfordQuestionAnsweringDataset =
                    StanfordQuestionAnsweringDataset.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optLimit(10)
                            .optUsage(Dataset.Usage.VALIDATION)
                            .build();

            stanfordQuestionAnsweringDataset.prepare();
            Assert.fail("Invalid data expects exception!");
        } catch (UnsupportedOperationException uoe) {
            Assert.assertEquals(uoe.getMessage(), "Validation data not available.");
        }
    }

    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/47
    @Test
    public void testScenario2() throws IOException, TranslateException {
        try (NDManager manager = NDManager.newBaseManager()) {
            StanfordQuestionAnsweringDataset stanfordQuestionAnsweringDataset =
                    StanfordQuestionAnsweringDataset.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optLimit(350)
                            .optUsage(Dataset.Usage.TEST)
                            .build();

            stanfordQuestionAnsweringDataset.prepare();
            stanfordQuestionAnsweringDataset.prepare();
            Assert.assertEquals(stanfordQuestionAnsweringDataset.size(), 350);

            Record record0 = stanfordQuestionAnsweringDataset.get(manager, 0);
            Record record6 = stanfordQuestionAnsweringDataset.get(manager, 6);
            Assert.assertEquals(record6.getData().get("title").getShape().dimension(), 2);
            Assert.assertEquals(
                    record0.getData().get("context").getShape().get(0),
                    record6.getData().get("context").getShape().get(0));
            Assert.assertEquals(record6.getLabels().size(), 0);
        }
    }
}
