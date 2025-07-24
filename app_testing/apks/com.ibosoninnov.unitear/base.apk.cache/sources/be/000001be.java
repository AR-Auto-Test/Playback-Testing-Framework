package b.b.h;

import android.content.Context;
import android.util.AttributeSet;
import android.view.ActionMode;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputConnection;
import android.widget.CheckedTextView;

/* compiled from: AppCompatCheckedTextView.java */
/* loaded from: classes.dex */
public class h extends CheckedTextView {

    /* renamed from: b  reason: collision with root package name */
    public static final int[] f843b = {16843016};

    /* renamed from: c  reason: collision with root package name */
    public final y f844c;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public h(Context context, AttributeSet attributeSet) {
        super(context, attributeSet, 16843720);
        v0.a(context);
        t0.a(this, getContext());
        y yVar = new y(this);
        this.f844c = yVar;
        yVar.e(attributeSet, 16843720);
        yVar.b();
        y0 r = y0.r(getContext(), attributeSet, f843b, 16843720, 0);
        setCheckMarkDrawable(r.g(0));
        r.f972b.recycle();
    }

    @Override // android.widget.CheckedTextView, android.widget.TextView, android.view.View
    public void drawableStateChanged() {
        super.drawableStateChanged();
        y yVar = this.f844c;
        if (yVar != null) {
            yVar.b();
        }
    }

    @Override // android.widget.TextView, android.view.View
    public InputConnection onCreateInputConnection(EditorInfo editorInfo) {
        InputConnection onCreateInputConnection = super.onCreateInputConnection(editorInfo);
        b.b.a.m(onCreateInputConnection, editorInfo, this);
        return onCreateInputConnection;
    }

    @Override // android.widget.CheckedTextView
    public void setCheckMarkDrawable(int i) {
        setCheckMarkDrawable(b.b.d.a.a.a(getContext(), i));
    }

    @Override // android.widget.TextView
    public void setCustomSelectionActionModeCallback(ActionMode.Callback callback) {
        super.setCustomSelectionActionModeCallback(b.j.b.d.U(this, callback));
    }

    @Override // android.widget.TextView
    public void setTextAppearance(Context context, int i) {
        super.setTextAppearance(context, i);
        y yVar = this.f844c;
        if (yVar != null) {
            yVar.f(context, i);
        }
    }
}