package c.e.b;

import android.util.Log;
import c.e.b.ef.a;
import com.ibosoninnov.unitear.ARGallerySubActivity;

/* compiled from: ARGallerySubActivity.java */
/* loaded from: classes2.dex */
public class vb implements a.b {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ ARGallerySubActivity f5332a;

    public vb(ARGallerySubActivity aRGallerySubActivity) {
        this.f5332a = aRGallerySubActivity;
    }

    @Override // c.e.b.ef.a.b
    public void a(final boolean z) {
        Log.d("ARGallery", "Empty " + z);
        this.f5332a.runOnUiThread(new Runnable() { // from class: c.e.b.g0
            @Override // java.lang.Runnable
            public final void run() {
                vb vbVar = vb.this;
                if (z) {
                    vbVar.f5332a.v.setVisibility(0);
                } else {
                    vbVar.f5332a.v.setVisibility(8);
                }
            }
        });
    }
}