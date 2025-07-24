package c.e.b.p000if;

import android.content.Context;
import android.graphics.Bitmap;
import android.text.TextUtils;
import android.util.Log;
import c.b.a.a.a;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

/* compiled from: Cache.java */
/* renamed from: c.e.b.if.e  reason: invalid package */
/* loaded from: classes2.dex */
public class e {

    /* renamed from: a  reason: collision with root package name */
    public static final String f4873a = "e";

    /* renamed from: b  reason: collision with root package name */
    public final Context f4874b;

    public e(Context context) {
        this.f4874b = context;
    }

    public String a(String str) {
        Log.i("gggggggggggggggggggggggggggggggggggg", "name   " + str);
        Context context = this.f4874b;
        if (!TextUtils.isEmpty(str)) {
            File file = new File(new File(context.getCacheDir(), "images"), str);
            if (file.exists()) {
                Log.i("gggggggggggggggggggggggggggggggggggg", "exits");
                return file.toString();
            }
            Log.i("gggggggggggggggggggggggggggggggggggg", "no there");
            return null;
        }
        Log.i("gggggggggggggggggggggggggggggggggggg", "return");
        return null;
    }

    public File b(Bitmap bitmap, String str) {
        File file;
        Log.i("gggggggggggggggggggggggggggggggggggg", "saving " + str);
        if (TextUtils.isEmpty(str)) {
            str = "img";
        }
        File file2 = null;
        try {
            file = new File(this.f4874b.getCacheDir(), "images");
            try {
                file.mkdirs();
                FileOutputStream fileOutputStream = new FileOutputStream(file + "/" + str);
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, fileOutputStream);
                fileOutputStream.close();
            } catch (IOException e2) {
                e = e2;
                file2 = file;
                StringBuilder x = a.x("error ");
                x.append(e.getMessage());
                Log.i("gggggggggggggggggggggggggggggggggggg", x.toString());
                String str2 = f4873a;
                Log.e(str2, "saveImgToCache error: " + bitmap, e);
                file = file2;
                Log.i("gggggggggggggggggggggggggggggggggggg", "cachePath " + file);
                return file;
            }
        } catch (IOException e3) {
            e = e3;
        }
        Log.i("gggggggggggggggggggggggggggggggggggg", "cachePath " + file);
        return file;
    }
}