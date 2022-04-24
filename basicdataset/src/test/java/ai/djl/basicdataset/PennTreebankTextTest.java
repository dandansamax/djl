/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.basicdataset.nlp.PennTreebankText;
import ai.djl.basicdataset.utils.TextData.Configuration;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Repository;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class PennTreebankTextTest {

    private static final int EMBEDDING_SIZE = 15;

    //CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/1579
    @Test
    public void testPennTreebankTextTrainLocal() throws IOException, TranslateException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
        try (NDManager manager = NDManager.newBaseManager()) {
            PennTreebankText dataset =
                    PennTreebankText.builder()
                            .setSourceConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE))
                                            .setEmbeddingSize(EMBEDDING_SIZE))
                            .setTargetConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE))
                                            .setEmbeddingSize(EMBEDDING_SIZE))
                            .setSampling(32, true)
                            .optLimit(100)
                            .optRepository(repository)
                            .optUsage(Dataset.Usage.TRAIN)
                            .build();

            dataset.prepare();
            Record record = dataset.get(manager, 0);
            Assert.assertEquals(record.getData().get(0).getShape().dimension(), 2);
            Assert.assertNull(record.getLabels());
        }
    }

    //CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/1579
    @Test
    public void testPennTreebankTextTestLocal() throws IOException, TranslateException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
        try (NDManager manager = NDManager.newBaseManager()) {
            PennTreebankText dataset =
                    PennTreebankText.builder()
                            .setSourceConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE))
                                            .setEmbeddingSize(EMBEDDING_SIZE))
                            .setTargetConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE))
                                            .setEmbeddingSize(EMBEDDING_SIZE))
                            .setSampling(32, true)
                            .optLimit(100)
                            .optRepository(repository)
                            .optUsage(Dataset.Usage.TEST)
                            .build();

            dataset.prepare();
            Record record = dataset.get(manager, 0);
            Assert.assertEquals(record.getData().get(0).getShape().dimension(), 2);
            Assert.assertNull(record.getLabels());
        }
    }

    //CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/1579
    @Test
    public void testPennTreebankTextValidationLocal() throws IOException, TranslateException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
        try (NDManager manager = NDManager.newBaseManager()) {
            PennTreebankText dataset =
                    PennTreebankText.builder()
                            .setSourceConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE))
                                            .setEmbeddingSize(EMBEDDING_SIZE))
                            .setTargetConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE))
                                            .setEmbeddingSize(EMBEDDING_SIZE))
                            .setSampling(32, true)
                            .optLimit(100)
                            .optRepository(repository)
                            .optUsage(Dataset.Usage.VALIDATION)
                            .build();

            dataset.prepare();
            Record record = dataset.get(manager, 0);
            Assert.assertEquals(record.getData().get(0).getShape().dimension(), 2);
            Assert.assertNull(record.getLabels());
        }
    }

    //CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/1579
    @Test
    public void testPrepare1() throws IOException, TranslateException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
        try (NDManager manager = NDManager.newBaseManager()) {
            PennTreebankText dataset =
                    PennTreebankText.builder()
                            .setSourceConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE))
                                            .setEmbeddingSize(EMBEDDING_SIZE))
                            .setTargetConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE))
                                            .setEmbeddingSize(EMBEDDING_SIZE))
                            .setSampling(32, true)
                            .optLimit(100)
                            .optRepository(repository)
                            .optUsage(Dataset.Usage.TRAIN)
                            .build();
            dataset.prepare();
        }
    }

    //CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/1579
    @Test
    public void testPrepare2() throws IOException, TranslateException {
        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
        try (NDManager manager = NDManager.newBaseManager()) {
            PennTreebankText dataset =
                    PennTreebankText.builder()
                            .setSourceConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE))
                                            .setEmbeddingSize(EMBEDDING_SIZE))
                            .setTargetConfiguration(
                                    new Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE))
                                            .setEmbeddingSize(EMBEDDING_SIZE))
                            .setSampling(32, true)
                            .optLimit(100)
                            .optRepository(repository)
                            .optUsage(Dataset.Usage.TEST)
                            .build();
            dataset.prepare();
        }
    }
}
