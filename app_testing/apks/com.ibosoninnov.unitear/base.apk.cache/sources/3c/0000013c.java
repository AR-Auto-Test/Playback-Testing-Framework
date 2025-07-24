package b.b.c;

import android.view.View;
import android.widget.AdapterView;
import androidx.appcompat.app.AlertController;

/* compiled from: AlertController.java */
/* loaded from: classes.dex */
public class e implements AdapterView.OnItemClickListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ AlertController f559b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ AlertController.b f560c;

    public e(AlertController.b bVar, AlertController alertController) {
        this.f560c = bVar;
        this.f559b = alertController;
    }

    @Override // android.widget.AdapterView.OnItemClickListener
    public void onItemClick(AdapterView<?> adapterView, View view, int i, long j) {
        this.f560c.w.onClick(this.f559b.f60b, i);
        if (this.f560c.G) {
            return;
        }
        this.f559b.f60b.dismiss();
    }
}