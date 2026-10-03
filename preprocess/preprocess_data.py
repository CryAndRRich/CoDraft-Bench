from config.config_data import CONFIG_DATA
def create_structured_text_enhanced(term, nature, purpose, class_id, nice_class_map):
    text = str(term).strip()
    context_parts = []
    if nature and str(nature).strip():
        context_parts.append(f"Nature: {str(nature).strip()}")
    if purpose and str(purpose).strip():
        context_parts.append(f"Use: {str(purpose).strip()}")
    if class_id:
        try:
            c_id = int(str(class_id))
            class_desc = nice_class_map.get(c_id, "")
            if class_desc:
                context_parts.append(f"Category: {class_desc}")
        except:
            pass
        
    if context_parts:
        full_text = f"{text} [ {' | '.join(context_parts)} ]"
    else:
        full_text = text
    return full_text

def preprocess(df):
    label_mapping = {
        'Dissimilar': 0,
        'Low similar': 1,
        'Similar': 2,
        'High similar': 3,
        'Identical': 4
    }
    df['label_score'] = df['Similarity'].map(label_mapping)
    df = df.dropna(subset=['label_score'])
    df['label_score'] = df['label_score'].astype(int)
    
    # The expanded variant carries the LLM-rewritten name instead of Nature/Purpose.
    if 'Term 1 Expand' in df.columns and 'Term 2 Expand' in df.columns:
        df['input_text_1'] = df['Term 1 Expand'].astype(str).str.strip()
        df['input_text_2'] = df['Term 2 Expand'].astype(str).str.strip()
        return df

    if 'Nature 1' in df.columns or 'Purpose 1' in df.columns:
        df['input_text_1'] = df.apply(
            lambda x: create_structured_text_enhanced(
                x['Term 1'], 
                x.get('Nature 1', ''),  
                x.get('Purpose 1', ''),
                x.get('Class 1', ''),
                CONFIG_DATA.NICE_CLASS_MAP
            ),
            axis=1
        )
    else:
        df['input_text_1'] = df['Term 1'].astype(str).str.strip()

    if 'Nature 2' in df.columns or 'Purpose 2' in df.columns:
        df['input_text_2'] = df.apply(
            lambda x: create_structured_text_enhanced(
                x['Term 2'], 
                x.get('Nature 2', ''), 
                x.get('Purpose 2', ''), 
                x.get('Class 2', ''), 
                CONFIG_DATA.NICE_CLASS_MAP
            ),
            axis=1
        )
    else:
        df['input_text_2'] = df['Term 2'].astype(str).str.strip()
        
    return df