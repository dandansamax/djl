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

import ai.djl.basicdataset.nlp.GoEmotions;
import ai.djl.basicdataset.utils.TextData;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Repository;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class GoEmotionsTest {

    private static final int EMBEDDING_SIZE = 15;

    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/1597
    @Test
    public void testPrepare1() throws IOException, TranslateException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
        try (NDManager manager = NDManager.newBaseManager()) {
            GoEmotions trainingSet =
                    GoEmotions.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .optRepository(repository)
                            .optUsage(Dataset.Usage.TRAIN)
                            .setSampling(32, true)
                            .build();
            trainingSet.prepare();

            long size = trainingSet.size();
            Assert.assertEquals(size, 43410);
        }
    }

    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/1597
    @Test
    public void testPrepare2() throws IOException, TranslateException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
        try (NDManager manager = NDManager.newBaseManager()) {
            GoEmotions trainingSet =
                    GoEmotions.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .optRepository(repository)
                            .optUsage(Dataset.Usage.TEST)
                            .setSampling(32, true)
                            .build();
            trainingSet.prepare();

            long size = trainingSet.size();
            Assert.assertEquals(size, 5427);
        }
    }

    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/1597
    @Test
    public void testGet1() throws IOException, TranslateException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
        try (NDManager manager = NDManager.newBaseManager()) {
            GoEmotions trainingSet =
                    GoEmotions.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .optRepository(repository)
                            .optUsage(Dataset.Usage.TEST)
                            .setSampling(32, true)
                            .build();
            trainingSet.prepare();

            long size = trainingSet.size();
            Assert.assertEquals(size, 5427);

            Record record = trainingSet.get(manager, 0);

            Assert.assertEquals(record.getData().size(), 1);
            Assert.assertEquals(record.getData().get(0).getShape().dimension(), 2);
            Assert.assertEquals(record.getLabels().size(), 1);
            Assert.assertEquals(record.getLabels().get(0).getShape().dimension(), 1);
        }
    }

    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/1597
    @Test
    public void testGet2() throws IOException, TranslateException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
        try (NDManager manager = NDManager.newBaseManager()) {
            GoEmotions trainingSet =
                    GoEmotions.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .optRepository(repository)
                            .optUsage(Dataset.Usage.VALIDATION)
                            .optLimit(10)
                            .setSampling(32, true)
                            .build();
            trainingSet.prepare();

            trainingSet.get(manager, 15);
            Assert.fail("Should fail at out-of-bound get!");
        } catch (IndexOutOfBoundsException exception) {
            Assert.assertTrue(exception.getMessage().contains("Index: 15, Size: 10"));
        }
    }

    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/1597
    @Test
    public void testScenario1() throws TranslateException, IOException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
        try (NDManager manager = NDManager.newBaseManager()) {
            GoEmotions trainingSet =
                    GoEmotions.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .optRepository(repository)
                            .optUsage(Dataset.Usage.TRAIN)
                            .setSampling(32, true)
                            .build();

            long size = trainingSet.size();
            Assert.assertEquals(size, 0);
        } catch (UnsupportedOperationException uoe) {
            Assert.assertEquals(uoe.getMessage(), "Validation data not available.");
        }
    }

    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/1597
    @Test
    public void testScenario2() throws IOException, TranslateException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
        try (NDManager manager = NDManager.newBaseManager()) {
            GoEmotions trainingSet =
                    GoEmotions.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .optRepository(repository)
                            .optUsage(Dataset.Usage.VALIDATION)
                            .setSampling(32, true)
                            .build();

            trainingSet.prepare();
            trainingSet.prepare();
            Assert.assertEquals(trainingSet.size(), 5426);

            Record record = trainingSet.get(manager, 0);

            Assert.assertEquals(record.getData().size(), 1);
            Assert.assertEquals(record.getData().get(0).getShape().dimension(), 2);
            Assert.assertEquals(record.getLabels().size(), 1);
            Assert.assertEquals(record.getLabels().get(0).getShape().dimension(), 1);
        }
    }
}
