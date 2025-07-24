package c.e.b;

import android.graphics.Bitmap;
import android.os.Handler;
import android.text.format.DateFormat;
import android.util.Log;
import android.view.PixelCopy;
import android.view.View;
import android.widget.Toast;
import com.ibosoninnov.unitear.NonARCoreActivitySceneform;
import com.ibosoninnov.unitear.R;
import java.io.File;
import java.io.FileOutputStream;
import java.util.Date;
import java.util.Objects;

/* compiled from: NonARCoreActivitySceneform.java */
/* loaded from: classes2.dex */
public class qe implements View.OnClickListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ NonARCoreActivitySceneform f5172b;

    public qe(NonARCoreActivitySceneform nonARCoreActivitySceneform) {
        this.f5172b = nonARCoreActivitySceneform;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        final NonARCoreActivitySceneform nonARCoreActivitySceneform = this.f5172b;
        if (nonARCoreActivitySceneform.X) {
            Toast.makeText(nonARCoreActivitySceneform.x, "Recording OFF", 1).show();
            NonARCoreActivitySceneform nonARCoreActivitySceneform2 = this.f5172b;
            nonARCoreActivitySceneform2.T.setText(nonARCoreActivitySceneform2.getResources().getString(R.string.photo_video));
            NonARCoreActivitySceneform.v(this.f5172b);
            return;
        }
        final Bitmap createBitmap = Bitmap.createBitmap(nonARCoreActivitySceneform.A.getArSceneView().getWidth(), nonARCoreActivitySceneform.A.getArSceneView().getHeight(), Bitmap.Config.ARGB_8888);
        PixelCopy.request(nonARCoreActivitySceneform.A.getArSceneView(), createBitmap, new PixelCopy.OnPixelCopyFinishedListener() { // from class: c.e.b.fb
            @Override // android.view.PixelCopy.OnPixelCopyFinishedListener
            public final void onPixelCopyFinished(int i) {
                String str;
                NonARCoreActivitySceneform nonARCoreActivitySceneform3 = NonARCoreActivitySceneform.this;
                Bitmap bitmap = createBitmap;
                Objects.requireNonNull(nonARCoreActivitySceneform3);
                if (i == 0) {
                    Log.d("NonARCoreActivity", "bitmapReady");
                    Date date = new Date();
                    DateFormat.format("yyyy-MM-dd_hh:mm:ss", date);
                    try {
                        str = nonARCoreActivitySceneform3.getCacheDir().getAbsolutePath() + "/" + date + ".jpg";
                        FileOutputStream fileOutputStream = new FileOutputStream(new File(str));
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 80, fileOutputStream);
                        fileOutputStream.flush();
                        fileOutputStream.close();
                    } catch (Throwable th) {
                        th.printStackTrace();
                        str = null;
                    }
                    bitmap.recycle();
                    if (str != null) {
                        nonARCoreActivitySceneform3.z(str, false);
                        return;
                    }
                    return;
                }
                Log.e("NonARCoreActivity", "captureImage error");
            }
        }, new Handler());
    }
}