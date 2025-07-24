package b.b.h;

import android.view.View;
import android.widget.AdapterView;

/* compiled from: ListPopupWindow.java */
/* loaded from: classes.dex */
public class j0 implements AdapterView.OnItemSelectedListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ k0 f871b;

    public j0(k0 k0Var) {
        this.f871b = k0Var;
    }

    @Override // android.widget.AdapterView.OnItemSelectedListener
    public void onItemSelected(AdapterView<?> adapterView, View view, int i, long j) {
        f0 f0Var;
        if (i == -1 || (f0Var = this.f871b.f876f) == null) {
            return;
        }
        f0Var.setListSelectionHidden(false);
    }

    @Override // android.widget.AdapterView.OnItemSelectedListener
    public void onNothingSelected(AdapterView<?> adapterView) {
    }
}