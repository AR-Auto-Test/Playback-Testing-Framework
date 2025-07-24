package b.k.a;

import android.database.Cursor;
import android.util.Log;
import android.widget.Filter;
import b.b.h.r0;
import java.util.Objects;

/* compiled from: CursorFilter.java */
/* loaded from: classes.dex */
public class b extends Filter {

    /* renamed from: a  reason: collision with root package name */
    public a f2313a;

    /* compiled from: CursorFilter.java */
    /* loaded from: classes.dex */
    public interface a {
    }

    public b(a aVar) {
        this.f2313a = aVar;
    }

    @Override // android.widget.Filter
    public CharSequence convertResultToString(Object obj) {
        return ((r0) this.f2313a).c((Cursor) obj);
    }

    /* JADX WARN: Removed duplicated region for block: B:20:0x0040  */
    /* JADX WARN: Removed duplicated region for block: B:21:0x0049  */
    @Override // android.widget.Filter
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public Filter.FilterResults performFiltering(CharSequence charSequence) {
        Cursor cursor;
        r0 r0Var = (r0) this.f2313a;
        Objects.requireNonNull(r0Var);
        String charSequence2 = charSequence == null ? "" : charSequence.toString();
        if (r0Var.n.getVisibility() == 0 && r0Var.n.getWindowVisibility() == 0) {
            try {
                cursor = r0Var.g(r0Var.o, charSequence2, 50);
            } catch (RuntimeException e2) {
                Log.w("SuggestionsAdapter", "Search suggestions query threw an exception.", e2);
            }
            if (cursor != null) {
                cursor.getCount();
                Filter.FilterResults filterResults = new Filter.FilterResults();
                if (cursor == null) {
                    filterResults.count = cursor.getCount();
                    filterResults.values = cursor;
                } else {
                    filterResults.count = 0;
                    filterResults.values = null;
                }
                return filterResults;
            }
        }
        cursor = null;
        Filter.FilterResults filterResults2 = new Filter.FilterResults();
        if (cursor == null) {
        }
        return filterResults2;
    }

    @Override // android.widget.Filter
    public void publishResults(CharSequence charSequence, Filter.FilterResults filterResults) {
        a aVar = this.f2313a;
        Cursor cursor = ((b.k.a.a) aVar).f2306d;
        Object obj = filterResults.values;
        if (obj == null || obj == cursor) {
            return;
        }
        ((r0) aVar).b((Cursor) obj);
    }
}