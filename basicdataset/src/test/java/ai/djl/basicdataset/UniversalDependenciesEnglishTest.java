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

import ai.djl.basicdataset.nlp.UniversalDependenciesEnglish;
import ai.djl.basicdataset.utils.TextData.Configuration;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Repository;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;

// CS304 Issue link: https://github.com/deepjavalibrary/djl/issues/46
public class UniversalDependenciesEnglishTest {

    private static final int EMBEDDING_SIZE = 15;

    @Test
    public void testPrepare1() throws IOException, TranslateException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
        try (NDManager manager = NDManager.newBaseManager()) {
            UniversalDependenciesEnglish universalDependenciesEnglish =
                    UniversalDependenciesEnglish.builder()
                            .optRepository(repository)
                            .optUsage(Dataset.Usage.TEST)
                            .setSourceConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optLimit(1211)
                            .build();

            universalDependenciesEnglish.prepare();
            Assert.assertEquals(universalDependenciesEnglish.size(), 1211);
        }
    }

    @Test
    public void testPrepare2() throws IOException, TranslateException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
        try (NDManager manager = NDManager.newBaseManager()) {
            UniversalDependenciesEnglish universalDependenciesEnglish =
                    UniversalDependenciesEnglish.builder()
                            .optRepository(repository)
                            .optUsage(Dataset.Usage.TEST)
                            .setSourceConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .build();

            universalDependenciesEnglish.prepare();
            Assert.assertEquals(universalDependenciesEnglish.size(), 2077);
        }
    }

    @Test
    public void testGet1() throws IOException, TranslateException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
        try (NDManager manager = NDManager.newBaseManager()) {
            UniversalDependenciesEnglish universalDependenciesEnglish =
                    UniversalDependenciesEnglish.builder()
                            .optRepository(repository)
                            .optUsage(Dataset.Usage.TEST)
                            .setSourceConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optLimit(1211)
                            .build();

            universalDependenciesEnglish.prepare();
            Record record = universalDependenciesEnglish.get(manager, 1210);
            Assert.assertEquals(record.getData().size(), 1);
            Assert.assertEquals(record.getData().get(0).getShape().dimension(), 2);
        }
    }

    @Test
    public void testGet2() throws IOException, TranslateException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
        try (NDManager manager = NDManager.newBaseManager()) {
            UniversalDependenciesEnglish universalDependenciesEnglish =
                    UniversalDependenciesEnglish.builder()
                            .optRepository(repository)
                            .optUsage(Dataset.Usage.TEST)
                            .setSourceConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optLimit(1027)
                            .build();

            universalDependenciesEnglish.prepare();
            Record record = universalDependenciesEnglish.get(manager, 1026);
            Assert.assertEquals(record.getLabels().size(), 1);
            Assert.assertEquals(record.getLabels().get(0).getShape().dimension(), 1);
        }
    }

    @Test
    public void testScenario1() throws IOException, TranslateException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
        try (NDManager manager = NDManager.newBaseManager()) {
            UniversalDependenciesEnglish universalDependenciesEnglish =
                    UniversalDependenciesEnglish.builder()
                            .optRepository(repository)
                            .optUsage(Dataset.Usage.TEST)
                            .setSourceConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optLimit(1210)
                            .build();

            universalDependenciesEnglish.prepare();
            // Out of preprocessed bound to get
            universalDependenciesEnglish.get(manager, 1210);
            Assert.fail("Should fail at out-of-bound get!");
        } catch (IndexOutOfBoundsException e) {
            Assert.assertTrue(e.getMessage().contains("Index: 1210, Size: 1210"));
        }
    }

    @Test
    public void testScenario2() throws IOException, TranslateException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
        try (NDManager manager = NDManager.newBaseManager()) {
            UniversalDependenciesEnglish universalDependenciesEnglish =
                    UniversalDependenciesEnglish.builder()
                            .optRepository(repository)
                            .optUsage(Dataset.Usage.TEST)
                            .setSourceConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optLimit(1211)
                            .build();

            universalDependenciesEnglish.prepare();
            universalDependenciesEnglish.prepare();
            Assert.assertEquals(universalDependenciesEnglish.size(), 1211);

            Record record520 = universalDependenciesEnglish.get(manager, 520);
            Record record1210 = universalDependenciesEnglish.get(manager, 1210);

            Assert.assertEquals(record520.getData().get(0).getShape().dimension(), 2);
            Assert.assertEquals(record1210.getData().get(0).getShape().dimension(), 2);

            Assert.assertEquals(record520.getLabels().get(0).getShape().dimension(), 1);
            Assert.assertEquals(record1210.getLabels().get(0).getShape().dimension(), 1);

            Assert.assertEquals(record520.getData().get(0).getShape().get(1), 15);
            Assert.assertEquals(
                    record520.getData().get(0).getShape().get(1),
                    record1210.getData().get(0).getShape().get(1));

            Assert.assertEquals(record520.getData().size(), 1);
            Assert.assertEquals(record1210.getData().size(), 1);

            Assert.assertEquals(record520.getLabels().size(), 1);
            Assert.assertEquals(record1210.getLabels().size(), 1);

            Assert.assertEquals(
                    record520.getData().get(0).getShape().get(0),
                    record520.getLabels().get(0).getShape().get(0));
            Assert.assertEquals(
                    record1210.getData().get(0).getShape().get(0),
                    record1210.getLabels().get(0).getShape().get(0));
        }
    }
}
