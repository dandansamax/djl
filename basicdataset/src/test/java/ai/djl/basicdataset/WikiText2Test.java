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

import ai.djl.basicdataset.nlp.WikiText2;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.Test;

public class WikiText2Test {
    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/45
    @Test
    public void testGetData1() throws IOException {
        WikiText2 trainingSet = WikiText2.builder().optUsage(Dataset.Usage.TRAIN).build();
        Path path = trainingSet.getData();
        Assert.assertTrue(Files.isRegularFile(path));
    }

    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/45
    @Test
    public void testGetData2() throws IOException {
        WikiText2 testSet = WikiText2.builder().optUsage(Dataset.Usage.TEST).build();
        Path path = testSet.getData();
        Assert.assertTrue(Files.isRegularFile(path));
    }

    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/45
    @Test
    public void testGetData3() throws IOException, TranslateException {
        try (NDManager manager = NDManager.newBaseManager()) {
            WikiText2 trainingSet = WikiText2.builder().optUsage(Dataset.Usage.TRAIN).build();
            Assert.assertEquals(trainingSet.getData(manager), null);
        }
    }

    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/45
    @Test
    public void testGetData4() throws IOException, TranslateException {
        try (NDManager manager = NDManager.newBaseManager()) {
            WikiText2 testSet = WikiText2.builder().optUsage(Dataset.Usage.TEST).build();
            Assert.assertEquals(testSet.getData(manager), null);
        }
    }

    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/45
    @Test
    public void testScenario1() throws IOException {
        WikiText2 trainingSet = WikiText2.builder().optUsage(Dataset.Usage.VALIDATION).build();
        Path path = trainingSet.getData();
        Assert.assertTrue(Files.isRegularFile(path));
        try (BufferedReader bufferedReader = Files.newBufferedReader(path)) {
            Assert.assertEquals(bufferedReader.readLine(), " ");
            Assert.assertEquals(bufferedReader.readLine(), " = Homarus gammarus = ");
        }
    }

    // CS304 (manually written) Issue link: https://github.com/deepjavalibrary/djl/issues/45
    @Test
    public void testScenario2() throws IOException, TranslateException {
        try (NDManager manager = NDManager.newBaseManager()) {
            WikiText2 validSet = WikiText2.builder().optUsage(Dataset.Usage.VALIDATION).build();
            Assert.assertThrows(
                    NullPointerException.class, () -> validSet.getData(manager).iterator());
        }
    }
}
