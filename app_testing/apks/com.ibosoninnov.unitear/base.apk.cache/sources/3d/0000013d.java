package b.b.c;

import android.view.View;
import android.widget.AdapterView;
import androidx.appcompat.app.AlertController;

/* compiled from: AlertController.java */
/* loaded from: classes.dex */
public class f implements AdapterView.OnItemClickListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ AlertController.RecycleListView f561b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ AlertController f562c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ AlertController.b f563d;

    public f(AlertController.b bVar, AlertController.RecycleListView recycleListView, AlertController alertController) {
        this.f563d = bVar;
        this.f561b = recycleListView;
        this.f562c = alertController;
    }

    @Override // android.widget.AdapterView.OnItemClickListener
    public void onItemClick(AdapterView<?> adapterView, View view, int i, long j) {
        boolean[] zArr = this.f563d.E;
        if (zArr != null) {
            zArr[i] = this.f561b.isItemChecked(i);
        }
        this.f563d.I.onClick(this.f562c.f60b, i, this.f561b.isItemChecked(i));
    }
}