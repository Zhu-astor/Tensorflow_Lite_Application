package com.android.example.cameraxapp

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.android.example.cameraxapp.databinding.ActivityMainBinding
import com.android.example.cameraxapp.ml.BestFloat32
import com.android.example.cameraxapp.ml.CompressedEsrgan
import com.android.example.cameraxapp.ml.Epoch300Float16

import com.android.example.cameraxapp.ml.Generator500
import com.android.example.cameraxapp.ml.Generator5002
import com.android.example.cameraxapp.ml.Generator5003
import org.tensorflow.lite.support.image.ColorSpaceType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.ByteArrayOutputStream
import java.io.InputStreamReader
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors



typealias LumaListener = (luma: Double) -> Unit

class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding

    private val activityResultLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions())
        { permissions ->
            // Handle Permission granted/rejected
            var permissionGranted = true
            permissions.entries.forEach {
                if (it.key in REQUIRED_PERMISSIONS && it.value == false)
                    permissionGranted = false
            }
            if (!permissionGranted) {
                Toast.makeText(baseContext,
                    "Permission request denied",
                    Toast.LENGTH_SHORT).show()
            } else {
                startCamera()

            }
        }
    private var imageCapture: ImageCapture? = null

    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions()
        }

        // Set up the listeners for take photo and video capture buttons
        viewBinding.imageCaptureButton.setOnClickListener { takePhoto() }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun takePhoto() {
        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return

        // Create time stamped name and MediaStore entry.
        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US)
            .format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if(Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/CameraX-Image")
            }
        }

        // Create output options object which contains file + metadata
        val outputOptions = ImageCapture.OutputFileOptions
            .Builder(contentResolver,
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues)
            .build()
        Log.d(TAG, MediaStore.Images.Media.EXTERNAL_CONTENT_URI.toString())
        // Set up image capture listener, which is triggered after photo has
        // been taken

        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {

                    val bitmap = imageProxyToBitmap(image)
                    val image3 = TensorImage.fromBitmap(bitmap)
                    saveProcessedImage(bitmap)
                    super.onCaptureSuccess(image)

//                  Different models to load
                    val model = Generator500.newInstance(this@MainActivity)
                    val model1 = CompressedEsrgan.newInstance(this@MainActivity)
                    val model2 = Generator5002.newInstance(this@MainActivity)
                    val model3 = Generator5003.newInstance(this@MainActivity)
                    val classification_model = BestFloat32.newInstance(this@MainActivity)
                    try {

                        val outputs = model3.process(image3)
                        Log.d("ModelProcess", "Outputs: ${outputs}")
                        val probabilityImage = outputs.probabilityAsTensorBuffer
                        printTensorBufferContent(probabilityImage)
                        val image5 = tensorBufferToTensorImage(probabilityImage)
                        printTensorImageContent(image5)


                        val h = image5.height
                        val w = image5.width
                        val bitmap3 = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                        val data = probabilityImage.floatArray

//                      [-1,1] -> [0,255]
                        for (y in 0 until h) {
                            for (x in 0 until w) {
                                val index = y * w + x
                                val r = ((data[index * 3] + 1) * 127.5f).toInt().coerceIn(0, 255)
                                val g = ((data[index * 3 + 1] + 1) * 127.5f).toInt().coerceIn(0, 255)
                                val b = ((data[index * 3 + 2] + 1) * 127.5f).toInt().coerceIn(0, 255)

                                val pixelValue = 0xFF shl 24 or (r shl 16) or (g shl 8) or b
                                bitmap3.setPixel(x, y, pixelValue)
                            }
                        }
                        //
                        val probabilityImageBitmap = bitmap3

                        model2.close()
                        saveProcessedImage(probabilityImageBitmap)
                        Log.d("ModelProcess", "The image had been process by reflection removal model then the image will input to classification model to evaluate the result")
                        //



                        val classification_image = TensorImage.fromBitmap(bitmap3)
                        val classification_results = classification_model.process(classification_image)
                        val classification_result = classification_results.outputAsCategoryList

                        Log.d("ModelProcess", "The results of classification model is :${classification_result.toString()}")
                        classification_model.close()
                    } catch (e: Exception) {
                        Log.e("ModelProcessError", "model.process 失敗: ${e.message}")
                    }


                }
                // Load the labels.txt file from assets
                fun loadLabels(context: Context, fileName: String): List<String> {
                    val labels = mutableListOf<String>()
                    context.assets.open(fileName).bufferedReader().useLines { lines ->
                        lines.forEach {
                            labels.add(it)
                        }
                    }
                    return labels
                }

                override fun onError(exception: ImageCaptureException) {
                    super.onError(exception)
                }
                fun printTensorImageContent(tensorImage: TensorImage) {
                    val tensorBuffer = tensorImage.tensorBuffer  // 将 TensorImage 转换为 TensorBuffer
                    printTensorBufferContent(tensorBuffer)       // 打印内容
                }
                fun printTensorBufferContent(tensorBuffer: TensorBuffer) {
                    val data = tensorBuffer.floatArray  // 获取浮点数组
                    val shape = tensorBuffer.shape      // 获取Tensor的形状
                    Log.d("TensorBufferShape", "Shape: ${shape.joinToString(", ")}")
                    // 仅打印前100个数据
                    for (i in data.indices.take(100)) {
                        Log.d("TensorBufferContent", "Index $i: ${data[i]}")
                    }
                }
                fun tensorBufferToTensorImage(tensorBuffer: TensorBuffer): TensorImage {
                    // 创建一个 TensorImage 对象，设置数据类型
                    val tensorImage = TensorImage(tensorBuffer.dataType)

                    // 使用适当的 ColorSpaceType 从 TensorBuffer 加载数据到 TensorImage
                    tensorImage.load(tensorBuffer, ColorSpaceType.RGB)

                    return tensorImage
                }
                fun imageProxyToBitmap(image: ImageProxy): Bitmap {
                    val yBuffer = image.planes[0].buffer  // Y plane
                    val uBuffer = image.planes[1].buffer  // U plane
                    val vBuffer = image.planes[2].buffer  // V plane

                    val ySize = yBuffer.remaining()
                    val uSize = uBuffer.remaining()
                    val vSize = vBuffer.remaining()

                    val nv21 = ByteArray(ySize + uSize + vSize)

                    // Copy Y, U, and V planes into the nv21 array
                    yBuffer.get(nv21, 0, ySize)
                    vBuffer.get(nv21, ySize, vSize)
                    uBuffer.get(nv21, ySize + vSize, uSize)

                    val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
                    val out = ByteArrayOutputStream()
                    yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
                    val imageBytes = out.toByteArray()

                    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                }

                private fun saveProcessedImage(processedBitmap: Bitmap) {
                    // 創建新的時間戳名稱和 MediaStore entry
                    val processedName = SimpleDateFormat(FILENAME_FORMAT, Locale.US)
                        .format(System.currentTimeMillis())
                    val processedContentValues = ContentValues().apply {
                        put(MediaStore.MediaColumns.DISPLAY_NAME, processedName)
                        put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                        if(Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/ProcessedImages")
                        }
                    }

                    // 將處理後的圖片插入到 MediaStore 並保存
                    val processedUri = contentResolver.insert(
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                        processedContentValues
                    )

                    processedUri?.let { uri ->
                        contentResolver.openOutputStream(uri)?.use { outputStream ->
                            processedBitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
                        }

                        // 成功後顯示消息
                        val msg = "Processed image saved: $uri"
                        Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                        Log.d(TAG, msg)
                    }
                }
            }
        )
    }



    @SuppressLint("RestrictedApi")
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()
                .setBufferFormat(ImageFormat.YUV_420_888)  // 指定为 YUV_420_888 格式
                .build()

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture)

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun requestPermissions() {
        activityResultLauncher.launch(REQUIRED_PERMISSIONS)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "CameraXApp"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private val REQUIRED_PERMISSIONS =
            mutableListOf (
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO
            ).apply {
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }
            }.toTypedArray()
    }
}