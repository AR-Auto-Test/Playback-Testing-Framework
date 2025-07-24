package c.e.b;

import android.util.Log;
import android.view.View;
import com.ibosoninnov.unitear.ARGallerySubActivity;

/* compiled from: ARGallerySubActivity.java */
/* loaded from: classes2.dex */
public class wb implements c.e.b.p000if.h {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ ARGallerySubActivity f5379a;

    public wb(ARGallerySubActivity aRGallerySubActivity) {
        this.f5379a = aRGallerySubActivity;
    }

    @Override // c.e.b.p000if.h
    public void a(View view, int i) {
        c.e.b.hf.b bVar = this.f5379a.u.f4704c.get(i);
        if (bVar != null) {
            Log.d("ARMenu", i + " " + bVar.category + " " + bVar.name + " " + bVar.glbFile);
            c.e.b.p000if.d dVar = this.f5379a.C;
            dVar.f4872b.putString("arGalleryFile", bVar.glbFile);
            dVar.f4872b.apply();
            c.e.b.p000if.d dVar2 = this.f5379a.C;
            dVar2.f4872b.putString("arGalleryFileId", bVar.id);
            dVar2.f4872b.apply();
            this.f5379a.finish();
        }
    }

    @Override // c.e.b.p000if.h
    public void b(View view, int i) {
    }
}