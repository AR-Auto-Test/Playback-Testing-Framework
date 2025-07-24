package c.e.b;

import android.app.Activity;
import android.app.Dialog;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import c.e.b.cc;
import com.ibosoninnov.unitear.ARGalleryActivity;
import com.ibosoninnov.unitear.R;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

/* compiled from: ARGalleryActivity.java */
/* loaded from: classes2.dex */
public class ub implements cc.a {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ ARGalleryActivity f5292a;

    public ub(ARGalleryActivity aRGalleryActivity) {
        this.f5292a = aRGalleryActivity;
    }

    @Override // c.e.b.cc.a
    public void a(String str) {
        ARGalleryActivity.s = null;
        this.f5292a.runOnUiThread(new Runnable() { // from class: c.e.b.y
            @Override // java.lang.Runnable
            public final void run() {
                ub ubVar = ub.this;
                ubVar.f5292a.D.setRefreshing(false);
                ubVar.f5292a.F.setVisibility(8);
                final ARGalleryActivity aRGalleryActivity = ubVar.f5292a;
                if (((Activity) aRGalleryActivity.t).isFinishing()) {
                    return;
                }
                final Dialog dialog = new Dialog(aRGalleryActivity, 16974402);
                dialog.setContentView(R.layout.no_internet);
                ((Button) dialog.findViewById(R.id.retryBtn)).setOnClickListener(new View.OnClickListener() { // from class: c.e.b.z
                    @Override // android.view.View.OnClickListener
                    public final void onClick(View view) {
                        ARGalleryActivity aRGalleryActivity2 = ARGalleryActivity.this;
                        Dialog dialog2 = dialog;
                        aRGalleryActivity2.w();
                        dialog2.dismiss();
                    }
                });
                dialog.show();
            }
        });
        Log.e("UnityPlayerActivity", str);
    }

    @Override // c.e.b.cc.a
    public void b(final String str) {
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File(this.f5292a.t.getFilesDir(), "storage.json")));
        bufferedWriter.write(str);
        bufferedWriter.close();
        c.e.b.p000if.d dVar = this.f5292a.E;
        dVar.f4872b.putBoolean("gallery_updated", true);
        dVar.f4872b.apply();
        this.f5292a.runOnUiThread(new Runnable() { // from class: c.e.b.x
            @Override // java.lang.Runnable
            public final void run() {
                ub ubVar = ub.this;
                String str2 = str;
                ARGalleryActivity aRGalleryActivity = ubVar.f5292a;
                aRGalleryActivity.A = str2;
                aRGalleryActivity.v(str2);
                ubVar.f5292a.D.setRefreshing(false);
                ubVar.f5292a.F.setVisibility(8);
            }
        });
    }
}