package c.e.b;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.provider.MediaStore;
import android.text.format.DateFormat;
import android.util.Log;
import android.widget.Toast;
import com.ibosoninnov.unitear.LoginWebviewActivity;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Date;
import java.util.Objects;

/* compiled from: LoginWebviewActivity.java */
/* loaded from: classes2.dex */
public class fe implements f.e {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ String f4756a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ LoginWebviewActivity f4757b;

    /* compiled from: LoginWebviewActivity.java */
    /* loaded from: classes2.dex */
    public class a implements Runnable {

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ Bitmap f4758b;

        public a(Bitmap bitmap) {
            this.f4758b = bitmap;
        }

        @Override // java.lang.Runnable
        public void run() {
            Bitmap bitmap = this.f4758b;
            if (bitmap != null) {
                fe feVar = fe.this;
                LoginWebviewActivity loginWebviewActivity = feVar.f4757b;
                String str = feVar.f4756a;
                Objects.requireNonNull(loginWebviewActivity);
                File file = new File(str);
                try {
                    CharSequence format = DateFormat.format("yyyy-MM-dd_hh:mm:ss", new Date());
                    String str2 = "IMG" + ((Object) format) + ".jpeg";
                    ContentResolver contentResolver = loginWebviewActivity.getContentResolver();
                    ContentValues contentValues = new ContentValues();
                    contentValues.put("_display_name", str2);
                    contentValues.put("mime_type", "image/jpeg");
                    contentValues.put("relative_path", str);
                    Uri insert = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues);
                    Objects.requireNonNull(insert);
                    OutputStream openOutputStream = contentResolver.openOutputStream(insert);
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 90, openOutputStream);
                    Objects.requireNonNull(openOutputStream);
                    openOutputStream.close();
                    loginWebviewActivity.v(file + "/" + str2);
                    return;
                } catch (Exception e2) {
                    Log.w("ExternalStorage", "Error writing " + str, e2);
                    Toast.makeText(loginWebviewActivity.getApplicationContext(), "Failed", 1).show();
                    return;
                }
            }
            Toast.makeText(fe.this.f4757b, "Failed to download", 1).show();
        }
    }

    public fe(LoginWebviewActivity loginWebviewActivity, String str) {
        this.f4757b = loginWebviewActivity;
        this.f4756a = str;
    }

    @Override // f.e
    public void a(f.d dVar, f.b0 b0Var) {
        if (b0Var.B()) {
            this.f4757b.runOnUiThread(new a(BitmapFactory.decodeStream(b0Var.f5730h.B())));
            return;
        }
        Log.e(LoginWebviewActivity.class.getName(), "download response unsucessfull");
    }

    @Override // f.e
    public void b(f.d dVar, IOException iOException) {
        String name = LoginWebviewActivity.class.getName();
        StringBuilder x = c.b.a.a.a.x("download ");
        x.append(iOException.toString());
        Log.e(name, x.toString());
    }
}