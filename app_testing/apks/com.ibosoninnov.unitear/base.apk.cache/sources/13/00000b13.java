package c.e.b;

import android.content.Intent;
import android.util.Log;
import android.view.View;
import com.ibosoninnov.unitear.ARGalleryActivity;
import com.ibosoninnov.unitear.ARGallerySubActivity;

/* compiled from: ARGalleryActivity.java */
/* loaded from: classes2.dex */
public class tb implements c.e.b.p000if.h {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ ARGalleryActivity f5256a;

    public tb(ARGalleryActivity aRGalleryActivity) {
        this.f5256a = aRGalleryActivity;
    }

    @Override // c.e.b.p000if.h
    public void a(View view, int i) {
        c.e.b.hf.b bVar = this.f5256a.z.f4704c.get(i);
        if (bVar != null) {
            Log.d("ARMenu", i + " " + bVar.category + " " + bVar.name + " " + bVar.glbFile);
            if (bVar.nameFilter == null) {
                Intent intent = new Intent(this.f5256a.t, ARGallerySubActivity.class);
                intent.putExtra("title", bVar.category);
                intent.putExtra("jsonData", this.f5256a.A);
                this.f5256a.overridePendingTransition(0, 0);
                intent.setFlags(65536);
                this.f5256a.startActivity(intent);
                return;
            }
            c.e.b.p000if.d dVar = this.f5256a.E;
            dVar.f4872b.putString("arGalleryFile", bVar.glbFile);
            dVar.f4872b.apply();
            c.e.b.p000if.d dVar2 = this.f5256a.E;
            dVar2.f4872b.putString("arGalleryFileId", bVar.id);
            dVar2.f4872b.apply();
            this.f5256a.finish();
        }
    }

    @Override // c.e.b.p000if.h
    public void b(View view, int i) {
    }
}