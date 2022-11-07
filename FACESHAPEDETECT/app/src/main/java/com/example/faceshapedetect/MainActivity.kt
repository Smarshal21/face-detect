package com.example.faceshapedetect
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import com.example.faceshapedetect.ml.ModelUnquant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.DequantizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import android.content.Intent as Intent1


class MainActivity : AppCompatActivity() {

    lateinit var select_image_button : Button
    lateinit var make_prediction : Button
    lateinit var img_view : ImageView
    lateinit var text_view : TextView
    lateinit var camerabtn : Button
    var uri:String = ""
    var bitmap:Bitmap? = null
    @RequiresApi(Build.VERSION_CODES.M)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        select_image_button = findViewById(R.id.button)
        make_prediction = findViewById(R.id.button2)
        img_view = findViewById(R.id.imageView2)
        text_view = findViewById(R.id.textView)
        camerabtn = findViewById<Button>(R.id.camerabtn)

        checkandGetpermissions()

        select_image_button.setOnClickListener(View.OnClickListener {
            Log.d("mssg", "button pressed")
            var intent : Intent1 = Intent1(Intent1.ACTION_GET_CONTENT)
            intent.type = "image/*"

            startActivityForResult(intent, 250)
        })

        camerabtn.setOnClickListener(View.OnClickListener {
            var camera : Intent1 = Intent1(android.provider.MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(camera, 200)
        })
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray){
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(requestCode == 100){
            if(grantResults[0] == PackageManager.PERMISSION_GRANTED)
            {
                Toast.makeText(this, "Camera permission granted", Toast.LENGTH_SHORT).show()
            }
            else{
                Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show()
            }
        }
    }
    @RequiresApi(Build.VERSION_CODES.M)

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent1?) {
        super.onActivityResult(requestCode, resultCode, data)

        if(requestCode == 250){
            img_view.setImageURI(data?.data)

            var uuri : Uri ?= data?.data
            uri = uuri.toString()
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uuri)
        }
        else if(requestCode == 200 && resultCode == Activity.RESULT_OK){
            bitmap = data?.extras?.get("data") as Bitmap
            img_view.setImageBitmap(bitmap)
        }
        fun ARGBBitmap(img: Bitmap): Bitmap {
            return img.copy(Bitmap.Config.ARGB_8888, true)
        }
        make_prediction.setOnClickListener(/* l = */ View.OnClickListener {
            val model = ModelUnquant.newInstance(this)
            if (bitmap != null) {
                val tbuffer:TensorImage = TensorImage.fromBitmap(ARGBBitmap(bitmap!!))
                var bitmapss:Bitmap = ARGBBitmap(bitmap!!)



                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
                val imageProcessor = ImageProcessor.Builder()
                    .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                    .build()
                var tensorImage = TensorImage(DataType.FLOAT32)
                tensorImage.load(bitmapss)
                tensorImage = imageProcessor.process(tensorImage)
                fun Bitmap.convertToByteArray(): ByteArray {
                    //minimum number of bytes that can be used to store this bitmap's pixels
                    val size = this.byteCount
                    //allocate new instances which will hold bitmap
                    val buffer = ByteBuffer.allocate(size)
                    val bytes = ByteArray(size)
                    //copy the bitmap's pixels into the specified buffer
                    this.copyPixelsToBuffer(buffer)
                    //rewinds buffer (buffer position is set to zero and the mark is discarded)
                    buffer.rewind()
                    //transfer bytes from buffer into the given destination array
                    buffer.get(bytes)
                    //return bitmap's pixels
                    return bytes
                }


                var probproc:TensorProcessor = TensorProcessor.Builder().add(DequantizeOp(0F,
                    (1/255.0).toFloat()
                )).build()
                var dequantbuff:TensorBuffer = probproc.process(inputFeature0)
                var buffer: ByteBuffer = tensorImage.buffer
                inputFeature0.loadBuffer(buffer)
                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer
                var confidence:FloatArray = outputFeature0.floatArray
                var Maxpos:Int = 0
                var i:Int
                var maxconfidence:Float = 0F
                for (i in 0..confidence.size-1){
                    if (confidence[i]>maxconfidence){
                        maxconfidence = confidence[i]
                        Maxpos = i
                    }
                }
                var classes = arrayListOf<String>("Heart","Oblong","Oval","Round","Square")
                text_view.setText(classes[Maxpos])
                model.close()

    }

})}

    @RequiresApi(Build.VERSION_CODES.M)
    public fun checkandGetpermissions(){
        if(checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED){
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 100)
        }
        else{
            Toast.makeText(this, "Camera permission granted", Toast.LENGTH_SHORT).show()
        }
    }

}