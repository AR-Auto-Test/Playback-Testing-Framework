package com.google.android.gms.common.data;

import android.content.ContentValues;
import android.database.CharArrayBuffer;
import android.database.Cursor;
import android.database.CursorIndexOutOfBoundsException;
import android.database.CursorWindow;
import android.os.Bundle;
import android.os.Parcel;
import android.os.Parcelable;
import android.util.Log;
import androidx.annotation.RecentlyNonNull;
import androidx.annotation.RecentlyNullable;
import com.google.android.gms.common.annotation.KeepForSdk;
import com.google.android.gms.common.annotation.KeepName;
import com.google.android.gms.common.internal.Asserts;
import com.google.android.gms.common.internal.Preconditions;
import com.google.android.gms.common.internal.safeparcel.AbstractSafeParcelable;
import com.google.android.gms.common.internal.safeparcel.SafeParcelWriter;
import com.google.android.gms.common.internal.safeparcel.SafeParcelable;
import com.google.android.gms.common.sqlite.CursorWrapper;
import java.io.Closeable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
@KeepForSdk
@KeepName
@SafeParcelable.Class(creator = "DataHolderCreator", validate = true)
/* loaded from: classes.dex */
public final class DataHolder extends AbstractSafeParcelable implements Closeable {
    @RecentlyNonNull
    @KeepForSdk
    public static final Parcelable.Creator<DataHolder> CREATOR = new zac();
    private static final Builder zak = new zab(new String[0], null);
    @SafeParcelable.VersionField(id = 1000)
    private final int zaa;
    @SafeParcelable.Field(getter = "getColumns", id = 1)
    private final String[] zab;
    private Bundle zac;
    @SafeParcelable.Field(getter = "getWindows", id = 2)
    private final CursorWindow[] zad;
    @SafeParcelable.Field(getter = "getStatusCode", id = 3)
    private final int zae;
    @SafeParcelable.Field(getter = "getMetadata", id = 4)
    private final Bundle zaf;
    private int[] zag;
    private int zah;
    private boolean zai;
    private boolean zaj;

    /* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
    @KeepForSdk
    /* loaded from: classes.dex */
    public static class Builder {
        private final String[] zaa;
        private final ArrayList<HashMap<String, Object>> zab;
        private final String zac;
        private final HashMap<Object, Integer> zad;
        private boolean zae;
        private String zaf;

        private Builder(String[] strArr, String str) {
            this.zaa = (String[]) Preconditions.checkNotNull(strArr);
            this.zab = new ArrayList<>();
            this.zac = null;
            this.zad = new HashMap<>();
            this.zae = false;
            this.zaf = null;
        }

        @RecentlyNonNull
        @KeepForSdk
        public DataHolder build(@RecentlyNonNull int i) {
            return new DataHolder(this, i, (Bundle) null, (zab) null);
        }

        @RecentlyNonNull
        @KeepForSdk
        public Builder withRow(@RecentlyNonNull ContentValues contentValues) {
            Asserts.checkNotNull(contentValues);
            HashMap<String, Object> hashMap = new HashMap<>(contentValues.size());
            for (Map.Entry<String, Object> entry : contentValues.valueSet()) {
                hashMap.put(entry.getKey(), entry.getValue());
            }
            return zaa(hashMap);
        }

        /* JADX WARN: Removed duplicated region for block: B:13:0x0031  */
        /* JADX WARN: Removed duplicated region for block: B:14:0x0037  */
        @RecentlyNonNull
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public Builder zaa(@RecentlyNonNull HashMap<String, Object> hashMap) {
            Object obj;
            int intValue;
            Asserts.checkNotNull(hashMap);
            String str = this.zac;
            if (str != null && (obj = hashMap.get(str)) != null) {
                Integer num = this.zad.get(obj);
                if (num == null) {
                    this.zad.put(obj, Integer.valueOf(this.zab.size()));
                } else {
                    intValue = num.intValue();
                    if (intValue != -1) {
                        this.zab.add(hashMap);
                    } else {
                        this.zab.remove(intValue);
                        this.zab.add(intValue, hashMap);
                    }
                    this.zae = false;
                    return this;
                }
            }
            intValue = -1;
            if (intValue != -1) {
            }
            this.zae = false;
            return this;
        }

        @RecentlyNonNull
        @KeepForSdk
        public DataHolder build(@RecentlyNonNull int i, @RecentlyNonNull Bundle bundle) {
            return new DataHolder(this, i, bundle, -1, (zab) null);
        }

        public /* synthetic */ Builder(String[] strArr, String str, zab zabVar) {
            this(strArr, null);
        }
    }

    /* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
    /* loaded from: classes.dex */
    public static class zaa extends RuntimeException {
        public zaa(String str) {
            super(str);
        }
    }

    @SafeParcelable.Constructor
    public DataHolder(@SafeParcelable.Param(id = 1000) int i, @SafeParcelable.Param(id = 1) String[] strArr, @SafeParcelable.Param(id = 2) CursorWindow[] cursorWindowArr, @SafeParcelable.Param(id = 3) int i2, @SafeParcelable.Param(id = 4) Bundle bundle) {
        this.zai = false;
        this.zaj = true;
        this.zaa = i;
        this.zab = strArr;
        this.zad = cursorWindowArr;
        this.zae = i2;
        this.zaf = bundle;
    }

    @RecentlyNonNull
    @KeepForSdk
    public static Builder builder(@RecentlyNonNull String[] strArr) {
        return new Builder(strArr, null, null);
    }

    @RecentlyNonNull
    @KeepForSdk
    public static DataHolder empty(@RecentlyNonNull int i) {
        return new DataHolder(zak, i, (Bundle) null);
    }

    @Override // java.io.Closeable, java.lang.AutoCloseable
    @KeepForSdk
    public final void close() {
        synchronized (this) {
            if (!this.zai) {
                this.zai = true;
                int i = 0;
                while (true) {
                    CursorWindow[] cursorWindowArr = this.zad;
                    if (i >= cursorWindowArr.length) {
                        break;
                    }
                    cursorWindowArr[i].close();
                    i++;
                }
            }
        }
    }

    public final void finalize() {
        try {
            if (this.zaj && this.zad.length > 0 && !isClosed()) {
                close();
                String obj = toString();
                StringBuilder sb = new StringBuilder(String.valueOf(obj).length() + 178);
                sb.append("Internal data leak within a DataBuffer object detected!  Be sure to explicitly call release() on all DataBuffer extending objects when you are done with them. (internal object: ");
                sb.append(obj);
                sb.append(")");
                Log.e("DataBuffer", sb.toString());
            }
        } finally {
            super.finalize();
        }
    }

    @RecentlyNonNull
    @KeepForSdk
    public final boolean getBoolean(@RecentlyNonNull String str, @RecentlyNonNull int i, @RecentlyNonNull int i2) {
        zaa(str, i);
        return Long.valueOf(this.zad[i2].getLong(i, this.zac.getInt(str))).longValue() == 1;
    }

    @RecentlyNonNull
    @KeepForSdk
    public final byte[] getByteArray(@RecentlyNonNull String str, @RecentlyNonNull int i, @RecentlyNonNull int i2) {
        zaa(str, i);
        return this.zad[i2].getBlob(i, this.zac.getInt(str));
    }

    @RecentlyNonNull
    @KeepForSdk
    public final int getCount() {
        return this.zah;
    }

    @RecentlyNonNull
    @KeepForSdk
    public final int getInteger(@RecentlyNonNull String str, @RecentlyNonNull int i, @RecentlyNonNull int i2) {
        zaa(str, i);
        return this.zad[i2].getInt(i, this.zac.getInt(str));
    }

    @RecentlyNonNull
    @KeepForSdk
    public final long getLong(@RecentlyNonNull String str, @RecentlyNonNull int i, @RecentlyNonNull int i2) {
        zaa(str, i);
        return this.zad[i2].getLong(i, this.zac.getInt(str));
    }

    @RecentlyNullable
    @KeepForSdk
    public final Bundle getMetadata() {
        return this.zaf;
    }

    @RecentlyNonNull
    @KeepForSdk
    public final int getStatusCode() {
        return this.zae;
    }

    @RecentlyNonNull
    @KeepForSdk
    public final String getString(@RecentlyNonNull String str, @RecentlyNonNull int i, @RecentlyNonNull int i2) {
        zaa(str, i);
        return this.zad[i2].getString(i, this.zac.getInt(str));
    }

    @RecentlyNonNull
    @KeepForSdk
    public final int getWindowIndex(@RecentlyNonNull int i) {
        int[] iArr;
        int i2 = 0;
        Preconditions.checkState(i >= 0 && i < this.zah);
        while (true) {
            iArr = this.zag;
            if (i2 >= iArr.length) {
                break;
            } else if (i < iArr[i2]) {
                i2--;
                break;
            } else {
                i2++;
            }
        }
        return i2 == iArr.length ? i2 - 1 : i2;
    }

    @RecentlyNonNull
    @KeepForSdk
    public final boolean hasColumn(@RecentlyNonNull String str) {
        return this.zac.containsKey(str);
    }

    @RecentlyNonNull
    @KeepForSdk
    public final boolean hasNull(@RecentlyNonNull String str, @RecentlyNonNull int i, @RecentlyNonNull int i2) {
        zaa(str, i);
        return this.zad[i2].isNull(i, this.zac.getInt(str));
    }

    @RecentlyNonNull
    @KeepForSdk
    public final boolean isClosed() {
        boolean z;
        synchronized (this) {
            z = this.zai;
        }
        return z;
    }

    @Override // android.os.Parcelable
    public final void writeToParcel(@RecentlyNonNull Parcel parcel, @RecentlyNonNull int i) {
        int beginObjectHeader = SafeParcelWriter.beginObjectHeader(parcel);
        SafeParcelWriter.writeStringArray(parcel, 1, this.zab, false);
        SafeParcelWriter.writeTypedArray(parcel, 2, this.zad, i, false);
        SafeParcelWriter.writeInt(parcel, 3, getStatusCode());
        SafeParcelWriter.writeBundle(parcel, 4, getMetadata(), false);
        SafeParcelWriter.writeInt(parcel, 1000, this.zaa);
        SafeParcelWriter.finishObjectHeader(parcel, beginObjectHeader);
        if ((i & 1) != 0) {
            close();
        }
    }

    public final void zaa() {
        this.zac = new Bundle();
        int i = 0;
        int i2 = 0;
        while (true) {
            String[] strArr = this.zab;
            if (i2 >= strArr.length) {
                break;
            }
            this.zac.putInt(strArr[i2], i2);
            i2++;
        }
        this.zag = new int[this.zad.length];
        int i3 = 0;
        while (true) {
            CursorWindow[] cursorWindowArr = this.zad;
            if (i < cursorWindowArr.length) {
                this.zag[i] = i3;
                i3 += this.zad[i].getNumRows() - (i3 - cursorWindowArr[i].getStartPosition());
                i++;
            } else {
                this.zah = i3;
                return;
            }
        }
    }

    @RecentlyNonNull
    public final double zab(@RecentlyNonNull String str, @RecentlyNonNull int i, @RecentlyNonNull int i2) {
        zaa(str, i);
        return this.zad[i2].getDouble(i, this.zac.getInt(str));
    }

    @KeepForSdk
    public DataHolder(@RecentlyNonNull String[] strArr, @RecentlyNonNull CursorWindow[] cursorWindowArr, @RecentlyNonNull int i, Bundle bundle) {
        this.zai = false;
        this.zaj = true;
        this.zaa = 1;
        this.zab = (String[]) Preconditions.checkNotNull(strArr);
        this.zad = (CursorWindow[]) Preconditions.checkNotNull(cursorWindowArr);
        this.zae = i;
        this.zaf = bundle;
        zaa();
    }

    private static CursorWindow[] zaa(CursorWrapper cursorWrapper) {
        int i;
        ArrayList arrayList = new ArrayList();
        try {
            int count = cursorWrapper.getCount();
            CursorWindow window = cursorWrapper.getWindow();
            if (window == null || window.getStartPosition() != 0) {
                i = 0;
            } else {
                window.acquireReference();
                cursorWrapper.setWindow(null);
                arrayList.add(window);
                i = window.getNumRows();
            }
            while (i < count) {
                if (!cursorWrapper.moveToPosition(i)) {
                    break;
                }
                CursorWindow window2 = cursorWrapper.getWindow();
                if (window2 != null) {
                    window2.acquireReference();
                    cursorWrapper.setWindow(null);
                } else {
                    window2 = new CursorWindow(false);
                    window2.setStartPosition(i);
                    cursorWrapper.fillWindow(i, window2);
                }
                if (window2.getNumRows() == 0) {
                    break;
                }
                arrayList.add(window2);
                i = window2.getStartPosition() + window2.getNumRows();
            }
            cursorWrapper.close();
            return (CursorWindow[]) arrayList.toArray(new CursorWindow[arrayList.size()]);
        } catch (Throwable th) {
            cursorWrapper.close();
            throw th;
        }
    }

    private DataHolder(CursorWrapper cursorWrapper, int i, Bundle bundle) {
        this(cursorWrapper.getColumnNames(), zaa(cursorWrapper), i, bundle);
    }

    @KeepForSdk
    public DataHolder(@RecentlyNonNull Cursor cursor, @RecentlyNonNull int i, Bundle bundle) {
        this(new CursorWrapper(cursor), i, bundle);
    }

    private DataHolder(Builder builder, int i, Bundle bundle) {
        this(builder.zaa, zaa(builder, -1), i, (Bundle) null);
    }

    private DataHolder(Builder builder, int i, Bundle bundle, int i2) {
        this(builder.zaa, zaa(builder, -1), i, bundle);
    }

    public /* synthetic */ DataHolder(Builder builder, int i, Bundle bundle, zab zabVar) {
        this(builder, i, (Bundle) null);
    }

    public /* synthetic */ DataHolder(Builder builder, int i, Bundle bundle, int i2, zab zabVar) {
        this(builder, i, bundle, -1);
    }

    private static CursorWindow[] zaa(Builder builder, int i) {
        if (builder.zaa.length == 0) {
            return new CursorWindow[0];
        }
        ArrayList arrayList = builder.zab;
        int size = arrayList.size();
        CursorWindow cursorWindow = new CursorWindow(false);
        ArrayList arrayList2 = new ArrayList();
        arrayList2.add(cursorWindow);
        cursorWindow.setNumColumns(builder.zaa.length);
        int i2 = 0;
        boolean z = false;
        while (i2 < size) {
            try {
                if (!cursorWindow.allocRow()) {
                    StringBuilder sb = new StringBuilder(72);
                    sb.append("Allocating additional cursor window for large data set (row ");
                    sb.append(i2);
                    sb.append(")");
                    Log.d("DataHolder", sb.toString());
                    cursorWindow = new CursorWindow(false);
                    cursorWindow.setStartPosition(i2);
                    cursorWindow.setNumColumns(builder.zaa.length);
                    arrayList2.add(cursorWindow);
                    if (!cursorWindow.allocRow()) {
                        Log.e("DataHolder", "Unable to allocate row to hold data.");
                        arrayList2.remove(cursorWindow);
                        return (CursorWindow[]) arrayList2.toArray(new CursorWindow[arrayList2.size()]);
                    }
                }
                Map map = (Map) arrayList.get(i2);
                boolean z2 = true;
                for (int i3 = 0; i3 < builder.zaa.length && z2; i3++) {
                    String str = builder.zaa[i3];
                    Object obj = map.get(str);
                    if (obj == null) {
                        z2 = cursorWindow.putNull(i2, i3);
                    } else if (obj instanceof String) {
                        z2 = cursorWindow.putString((String) obj, i2, i3);
                    } else if (obj instanceof Long) {
                        z2 = cursorWindow.putLong(((Long) obj).longValue(), i2, i3);
                    } else if (obj instanceof Integer) {
                        z2 = cursorWindow.putLong(((Integer) obj).intValue(), i2, i3);
                    } else if (obj instanceof Boolean) {
                        z2 = cursorWindow.putLong(((Boolean) obj).booleanValue() ? 1L : 0L, i2, i3);
                    } else if (obj instanceof byte[]) {
                        z2 = cursorWindow.putBlob((byte[]) obj, i2, i3);
                    } else if (obj instanceof Double) {
                        z2 = cursorWindow.putDouble(((Double) obj).doubleValue(), i2, i3);
                    } else if (obj instanceof Float) {
                        z2 = cursorWindow.putDouble(((Float) obj).floatValue(), i2, i3);
                    } else {
                        String valueOf = String.valueOf(obj);
                        StringBuilder sb2 = new StringBuilder(String.valueOf(str).length() + 32 + valueOf.length());
                        sb2.append("Unsupported object for column ");
                        sb2.append(str);
                        sb2.append(": ");
                        sb2.append(valueOf);
                        throw new IllegalArgumentException(sb2.toString());
                    }
                }
                if (z2) {
                    z = false;
                } else if (!z) {
                    StringBuilder sb3 = new StringBuilder(74);
                    sb3.append("Couldn't populate window data for row ");
                    sb3.append(i2);
                    sb3.append(" - allocating new window.");
                    Log.d("DataHolder", sb3.toString());
                    cursorWindow.freeLastRow();
                    cursorWindow = new CursorWindow(false);
                    cursorWindow.setStartPosition(i2);
                    cursorWindow.setNumColumns(builder.zaa.length);
                    arrayList2.add(cursorWindow);
                    i2--;
                    z = true;
                } else {
                    throw new zaa("Could not add the value to a new CursorWindow. The size of value may be larger than what a CursorWindow can handle.");
                }
                i2++;
            } catch (RuntimeException e2) {
                int size2 = arrayList2.size();
                for (int i4 = 0; i4 < size2; i4++) {
                    ((CursorWindow) arrayList2.get(i4)).close();
                }
                throw e2;
            }
        }
        return (CursorWindow[]) arrayList2.toArray(new CursorWindow[arrayList2.size()]);
    }

    private final void zaa(String str, int i) {
        Bundle bundle = this.zac;
        if (bundle != null && bundle.containsKey(str)) {
            if (!isClosed()) {
                if (i < 0 || i >= this.zah) {
                    throw new CursorIndexOutOfBoundsException(i, this.zah);
                }
                return;
            }
            throw new IllegalArgumentException("Buffer is closed.");
        }
        String valueOf = String.valueOf(str);
        throw new IllegalArgumentException(valueOf.length() != 0 ? "No such column: ".concat(valueOf) : new String("No such column: "));
    }

    @RecentlyNonNull
    public final float zaa(@RecentlyNonNull String str, @RecentlyNonNull int i, @RecentlyNonNull int i2) {
        zaa(str, i);
        return this.zad[i2].getFloat(i, this.zac.getInt(str));
    }

    public final void zaa(@RecentlyNonNull String str, @RecentlyNonNull int i, @RecentlyNonNull int i2, @RecentlyNonNull CharArrayBuffer charArrayBuffer) {
        zaa(str, i);
        this.zad[i2].copyStringToBuffer(i, this.zac.getInt(str), charArrayBuffer);
    }
}