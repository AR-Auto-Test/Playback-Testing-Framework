package b.b.c;

import android.content.Context;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import androidx.appcompat.app.AlertController;

/* compiled from: AlertController.java */
/* loaded from: classes.dex */
public class c extends ArrayAdapter<CharSequence> {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ AlertController.RecycleListView f552b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ AlertController.b f553c;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public c(AlertController.b bVar, Context context, int i, int i2, CharSequence[] charSequenceArr, AlertController.RecycleListView recycleListView) {
        super(context, i, i2, charSequenceArr);
        this.f553c = bVar;
        this.f552b = recycleListView;
    }

    @Override // android.widget.ArrayAdapter, android.widget.Adapter
    public View getView(int i, View view, ViewGroup viewGroup) {
        View view2 = super.getView(i, view, viewGroup);
        boolean[] zArr = this.f553c.E;
        if (zArr != null && zArr[i]) {
            this.f552b.setItemChecked(i, true);
        }
        return view2;
    }
}