package b.b.c;

import android.content.Context;
import android.database.Cursor;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckedTextView;
import android.widget.CursorAdapter;
import androidx.appcompat.app.AlertController;

/* compiled from: AlertController.java */
/* loaded from: classes.dex */
public class d extends CursorAdapter {

    /* renamed from: b  reason: collision with root package name */
    public final int f554b;

    /* renamed from: c  reason: collision with root package name */
    public final int f555c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ AlertController.RecycleListView f556d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ AlertController f557e;

    /* renamed from: f  reason: collision with root package name */
    public final /* synthetic */ AlertController.b f558f;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public d(AlertController.b bVar, Context context, Cursor cursor, boolean z, AlertController.RecycleListView recycleListView, AlertController alertController) {
        super(context, cursor, z);
        this.f558f = bVar;
        this.f556d = recycleListView;
        this.f557e = alertController;
        Cursor cursor2 = getCursor();
        this.f554b = cursor2.getColumnIndexOrThrow(bVar.K);
        this.f555c = cursor2.getColumnIndexOrThrow(bVar.L);
    }

    @Override // android.widget.CursorAdapter
    public void bindView(View view, Context context, Cursor cursor) {
        ((CheckedTextView) view.findViewById(16908308)).setText(cursor.getString(this.f554b));
        this.f556d.setItemChecked(cursor.getPosition(), cursor.getInt(this.f555c) == 1);
    }

    @Override // android.widget.CursorAdapter
    public View newView(Context context, Cursor cursor, ViewGroup viewGroup) {
        return this.f558f.f71b.inflate(this.f557e.M, viewGroup, false);
    }
}